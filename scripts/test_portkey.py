"""Quick sanity-check for Portkey connectivity.

Usage:
    python scripts/test_portkey.py --api-key pk-xxx [--virtual-key vk-xxx] [--model claude-opus-4-7]
"""
import argparse, os, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("PORTKEY_API_KEY", ""))
    p.add_argument("--virtual-key", default=os.environ.get("PORTKEY_VIRTUAL_KEY", ""))
    p.add_argument("--model", default="claude-opus-4-7")
    args = p.parse_args()

    if not args.api_key:
        sys.exit("ERROR: provide --api-key or set PORTKEY_API_KEY env var")

    print(f"api_key  : {args.api_key[:8]}...")
    print(f"virt_key : {args.virtual_key[:8] + '...' if args.virtual_key else '(none)'}")
    print(f"model    : {args.model}")
    print("Sending test request...")

    from portkey_ai import Portkey
    kwargs = {"api_key": args.api_key}
    if args.virtual_key:
        kwargs["virtual_key"] = args.virtual_key
    client = Portkey(**kwargs)

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
            max_tokens=10,
        )
        print("SUCCESS:", resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"FAILED ({type(e).__name__}): {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
