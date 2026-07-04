import unittest

from src.app_errors import format_exception_for_user, sanitize_error_text


class AppErrorTests(unittest.TestCase):
    def test_sanitize_error_text_redacts_api_keys(self):
        message = "Invalid key sk-proj-abc1234567890SECRETKEY"

        sanitized = sanitize_error_text(message)

        self.assertNotIn("sk-proj-abc1234567890SECRETKEY", sanitized)
        self.assertIn("[redacted-api-key]", sanitized)

    def test_format_exception_for_user_classifies_model_errors(self):
        error = ValueError("The model `missing-model` does not exist")

        user_error = format_exception_for_user(error, "Generating answer")

        self.assertIn("model", user_error.title.lower())
        self.assertIn("OPENAI_MODEL", user_error.suggestion)


if __name__ == "__main__":
    unittest.main()
