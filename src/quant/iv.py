import math
from typing import Optional, Literal
from .pricing import bs_price


def _bounds_call_put(
    S: float, K: float, T: float, r: float, option_type: str
) -> tuple[float, float]:
    """Return (lower, upper) no-arbitrage bounds for a European option."""
    df = math.exp(-r * T) if T > 0 else 1.0
    if option_type == "call":
        lower = max(0.0, S - K * df)
        upper = S
    else:
        lower = max(0.0, K * df - S)
        upper = K * df
    return lower, upper


def _brent_root_finding(f, a, b, tol=1e-8, maxiter=100):
    """
    Brent's method for finding root of f(x) = 0 in [a, b].
    Assumes f(a) and f(b) have opposite signs.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Brent's method requires f(a) and f(b) to have opposite signs")

    # Ensure |f(b)| <= |f(a)|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for i in range(maxiter):
        if abs(fb) <= tol:
            return b

        if abs(fa) <= tol:
            return a

        # Check if we're close enough
        tol1 = 2 * tol * abs(b) + tol
        xm = 0.5 * (a - b)

        if abs(xm) <= tol1 or abs(fb) < tol:
            return b

        # Try inverse quadratic interpolation
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Linear interpolation
                p = 2 * xm * s
                q = 1 - s
            else:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)

            if p > 0:
                q = -q
            p = abs(p)

            # Check if interpolation is acceptable
            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        # Update a and b
        a = b
        fa = fb

        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm >= 0 else -tol1

        fb = f(b)

        if fb == 0:
            return b

        # Update c if needed
        if fb * fc > 0:
            c = a
            fc = fa
            d = b - a
            e = d
        else:
            if abs(fc) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
                c = a
                fc = fa
                d = b - a
                e = d

    # Return best approximation
    if abs(fa) < abs(fb):
        return a
    else:
        return b


def _bisection_root_finding(f, a, b, tol=1e-8, maxiter=100):
    """
    Robust bisection method for finding root of f(x) = 0 in [a, b].
    Assumes f(a) and f(b) have opposite signs.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Bisection requires f(a) and f(b) to have opposite signs")

    for i in range(maxiter):
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) <= tol or abs(b - a) <= tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (a + b) / 2


def _find_bracket(f, a, b, max_expansions=10):
    """
    Find a bracket [a, b] where f(a) and f(b) have opposite signs.
    Expand the interval if needed.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb < 0:
        return a, b

    # Try expanding the interval
    for i in range(max_expansions):
        # Expand by factor of 2
        a_new = a / 2
        b_new = b * 2

        fa_new = f(a_new)
        fb_new = f(b_new)

        if fa_new * fb_new < 0:
            return a_new, b_new

        if fa * fa_new < 0:
            return a_new, a

        if fb * fb_new < 0:
            return b, b_new

        # If the function is very flat, try a wider range
        if abs(fa_new - fa) < 1e-10 and abs(fb_new - fb) < 1e-10:
            # Try a much wider range
            a_wide = 1e-10
            b_wide = 10.0
            fa_wide = f(a_wide)
            fb_wide = f(b_wide)

            if fa_wide * fb_wide < 0:
                return a_wide, b_wide

        a, b = a_new, b_new
        fa, fb = fa_new, fb_new

    raise ValueError("Could not find bracket with sign change")


def implied_vol_bs(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-8,
    maxiter: int = 100,
    bracket: Optional[tuple[float, float]] = (1e-6, 5.0),
) -> float:
    """
    Solve for sigma so bs_price(S,K,T,r,sigma,option_type) == price.
    Use Brent if possible; otherwise robust bisection. Enforce no-arb bounds.
    For T==0 or price==intrinsic within 1e-12, return 0.0.
    """
    # Check for edge cases
    if T <= 0:
        return 0.0

    # Check no-arbitrage bounds
    lower, upper = _bounds_call_put(S, K, T, r, option_type)

    # Check if price is at intrinsic value (within tolerance)
    if abs(price - lower) < 1e-12:
        return 0.0

    # Check if price violates bounds
    if price < lower - 1e-12 or price > upper + 1e-12:
        raise ValueError(
            f"Price {price:.6f} violates no-arbitrage bounds "
            f"[{lower:.6f}, {upper:.6f}] for {option_type} option"
        )

    # Define the function to find root of: f(sigma) = bs_price(sigma) - price
    def f(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price

    # Use provided bracket or default
    a, b = bracket

    # Ensure bracket is valid
    if a <= 0 or b <= 0 or a >= b:
        raise ValueError(f"Invalid bracket [{a}, {b}]. Must have 0 < a < b")

    try:
        # Try to find a valid bracket
        a, b = _find_bracket(f, a, b)

        # Check if we already have a root at the bounds
        fa, fb = f(a), f(b)
        if abs(fa) <= tol:
            return a
        if abs(fb) <= tol:
            return b

        # If one of the bounds is very close to zero, return it
        if abs(fa) < 1e-10:
            return a
        if abs(fb) < 1e-10:
            return b

        # Check if the function is very flat in the lower part
        # (indicating low volatility)
        # Test a few points in the lower range
        test_sigmas = [a, a * 10, a * 100, a * 1000]
        test_values = [f(sigma) for sigma in test_sigmas]
        if (
            all(abs(val - test_values[0]) < 1e-8 for val in test_values)
            and abs(test_values[0]) < 1e-6
        ):
            # Use grid search for low volatility cases
            sigmas = [
                1e-10,
                1e-8,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                0.1,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
            ]
            for sigma in sigmas:
                if abs(f(sigma)) <= tol:
                    return sigma

        # Try Brent's method first
        try:
            return _brent_root_finding(f, a, b, tol, maxiter)
        except (ValueError, RuntimeError):
            # Fallback to bisection
            return _bisection_root_finding(f, a, b, tol, maxiter)

    except ValueError as e:
        # If we can't find a bracket, try a grid search
        # This handles cases where the function is very flat
        sigmas = [
            1e-10,
            1e-8,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ]
        for sigma in sigmas:
            if abs(f(sigma)) <= tol:
                return sigma

        # If grid search fails, check if price is very close to intrinsic
        if abs(price - lower) < 1e-10:
            return 0.0
        else:
            raise ValueError(f"Could not find implied volatility: {e}")
