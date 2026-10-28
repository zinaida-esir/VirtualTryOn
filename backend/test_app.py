import os

# Use a temporary database for testing
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key"

import pytest

from app import User, app, db


# Prepare the application before every test
@pytest.fixture()
def client():
    app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret-key",
    )

    # Create an empty database for the current test
    with app.app_context():
        db.session.remove()
        db.drop_all()
        db.create_all()

    with app.test_client() as test_client:
        yield test_client

    # Remove test data after the test is finished
    with app.app_context():
        db.session.remove()
        db.drop_all()


# Test 1: registration of a new user
def test_user_registration(client):
    response = client.post(
        "/api/register",
        json={
            "email": "student@example.com",
            "password": "Password123",
        },
    )

    # The server should return successful registration
    assert response.status_code == 201

    data = response.get_json()

    assert data["message"] == "Registration successful."
    assert data["user"]["email"] == "student@example.com"

    # Check that the user was added to the database
    with app.app_context():
        user = User.query.filter_by(
            email="student@example.com"
        ).first()

        assert user is not None


# Test 2: registration with the same email
def test_duplicate_email_is_rejected(client):
    # First registration
    first_response = client.post(
        "/api/register",
        json={
            "email": "student@example.com",
            "password": "Password123",
        },
    )

    # Second registration with the same email
    second_response = client.post(
        "/api/register",
        json={
            "email": "student@example.com",
            "password": "DifferentPassword456",
        },
    )

    assert first_response.status_code == 201

    # The second registration should be rejected
    assert second_response.status_code == 409

    data = second_response.get_json()

    assert data["error"] == "This email is already registered."

    # The database should still contain only one user
    with app.app_context():
        users_count = User.query.filter_by(
            email="student@example.com"
        ).count()

        assert users_count == 1


# Test 3: access to wardrobe without login
def test_wardrobe_requires_login(client):
    response = client.get("/api/results")

    # An unauthorised user should not get access
    assert response.status_code == 401

    data = response.get_json()

    assert data["error"] == "Authentication required."