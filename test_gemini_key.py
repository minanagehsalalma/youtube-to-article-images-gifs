import os


def main() -> int:
    # The SDK reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
    if not (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')):
        print('Missing GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.')
        return 2

    try:
        from google import genai
    except Exception as e:
        print('google-genai is not installed. Run: pip install -U google-genai')
        print(f'Details: {e}')
        return 2

    try:
        client = genai.Client()
        # 1) Lightweight auth check: list a couple models
        print('Listing a few available models...')
        for i, model in enumerate(client.models.list()):
            print(f'  - {getattr(model, "name", model)}')
            if i >= 4:
                break

        # 2) Simple generation
        response = client.models.generate_content(
            model=os.getenv('GEMINI_TEST_MODEL', 'gemini-2.5-flash'),
            contents='Reply with exactly: OK',
        )
        print('Generation response:', (response.text or '').strip())
        client.close()
        return 0
    except Exception as e:
        print('Gemini request failed.')
        print('If you see 401/403: key is invalid or restricted. If 404: model name not available.')
        print(f'Details: {e}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
