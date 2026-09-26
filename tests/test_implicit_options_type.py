#!/usr/bin/env python3
"""ImplicitOptions.type() names the scheme the options hold, and cannot be set.

  python test_implicit_options_type.py
"""
import sys

from snapy import ImplicitOptions


def main():
    failures = []
    for scheme, name in ((0, "none"), (1, "vic-partial"), (9, "vic-full")):
        op = ImplicitOptions()
        op.scheme(scheme)
        try:
            got = op.type()
        except Exception as e:  # noqa: BLE001
            got = f"raised {type(e).__name__}: {e}"
        if got != name:
            failures.append(f"scheme {scheme}: type() = {got!r}, expected {name!r}")

    op = ImplicitOptions()
    op.scheme(5)
    try:
        got = op.type()
        failures.append(f"scheme 5: type() = {got!r}, expected an error")
    except RuntimeError:
        pass
    except Exception as e:  # noqa: BLE001
        failures.append(f"scheme 5: type() raised {type(e).__name__}: {e}, expected RuntimeError")

    try:
        ImplicitOptions().type("vic-full")
        failures.append("type('vic-full') was accepted; type is read-only")
    except TypeError:
        pass
    except Exception as e:  # noqa: BLE001
        failures.append(f"type('vic-full') raised {type(e).__name__}: {e}, expected TypeError")

    for f in failures:
        print("FAIL", f)
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
