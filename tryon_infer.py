# tryon_infer.py
# coding: utf-8
import torch
import torch.nn.functional as F

from networks import GMM, UnetGenerator, load_checkpoint


class TryOnModel:
    """
    Минимальная обёртка для inference CP-VTON+ (GMM + TOM).
    """

    def __init__(
        self,
        gmm_ckpt: str,
        tom_ckpt: str,
        device: str = None,
        use_fp16: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and (self.device.startswith("cuda"))

        # ---- GMM ----
        # GMM обычно требует opt, но во многих реализациях он читает только пару полей.
        # Если твой GMM(opt) строго требует opt, ниже дам вариант как сделать "мини-opt".
        self.gmm = self._build_gmm(gmm_ckpt)

        # ---- TOM ----
        # CP-VTON+: входов 26 каналов, выход 6 (p_rendered(3) + m_composite(3))
        self.tom = UnetGenerator(26, 4, 6, ngf=64, norm_layer=torch.nn.InstanceNorm2d)
        load_checkpoint(self.tom, tom_ckpt)
        self.tom.to(self.device).eval()

    def _build_gmm(self, gmm_ckpt: str):
        # Если твой GMM требует opt с полями (fine_width, fine_height, grid_size и т.п.)
        # то подставляем "мини-объект" вместо argparse.
        class _Opt:
            fine_width = 192
            fine_height = 256
            grid_size = 5
            radius = 5

        gmm = GMM(_Opt())
        load_checkpoint(gmm, gmm_ckpt)
        gmm.to(self.device).eval()
        return gmm

    @torch.inference_mode()
    def warp_cloth(self, cloth, cloth_mask, agnostic):
        """
        Входы: Tensor NCHW в диапазоне [-1, 1]
        cloth:      [N,3,H,W]
        cloth_mask: [N,1,H,W] (0..1 или -1..1 — лучше 0..1)
        agnostic:   [N,C,H,W] (как в твоей подготовке)
        Выходы: warped_cloth [N,3,H,W], warped_mask [N,1,H,W]
        """
        cloth = cloth.to(self.device)
        cloth_mask = cloth_mask.to(self.device)
        agnostic = agnostic.to(self.device)

        if self.use_fp16:
            cloth = cloth.half()
            cloth_mask = cloth_mask.half()
            agnostic = agnostic.half()
            self.gmm.half()

        grid, _theta = self.gmm(agnostic, cloth_mask)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode="border", align_corners=False)
        warped_mask = F.grid_sample(cloth_mask, grid, padding_mode="zeros", align_corners=False)
        return warped_cloth, warped_mask

    @torch.inference_mode()
    def tryon(self, agnostic, cloth, cloth_mask):
        """
        Полный try-on: GMM warp -> TOM композит.
        Входы: Tensor NCHW в [-1,1], кроме маски (желательно 0..1).
        Выход: p_tryon [N,3,H,W] в [-1,1]
        """
        warped_cloth, warped_mask = self.warp_cloth(cloth, cloth_mask, agnostic)

        # TOM ждёт: cat([agnostic, warped_cloth, warped_mask], dim=1)
        x = torch.cat([agnostic, warped_cloth, warped_mask], dim=1).to(self.device)

        if self.use_fp16:
            x = x.half()
            self.tom.half()

        outputs = self.tom(x)

        p_rendered, m_composite = torch.split(outputs, 3, dim=1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)

        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        return p_tryon

    @torch.inference_mode()
    def tryon_return_uint8(self, agnostic, cloth, cloth_mask):
        """
        То же самое, но вернёт uint8 [N,H,W,3] готовое для веба.
        """
        out = self.tryon(agnostic, cloth, cloth_mask)  # [-1,1]
        out = (out.clamp(-1, 1) + 1) * 0.5  # [0,1]
        out = (out * 255.0).round().byte()
        out = out.permute(0, 2, 3, 1).contiguous()  # NHWC
        return out
