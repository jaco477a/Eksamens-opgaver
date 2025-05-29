from scipy.integrate import quad
from scipy.stats import beta, uniform, gamma  # Du kan også importere andre fordelinger

# Eksempel: F_cdf baseret på en Beta(3,2)-fordeling
F_cdf_beta = lambda v: beta.cdf(v, 3, 2)

# Antal observationer (skal defineres før kald af b)
n = 5  # eller hvad din n skal være

def b(v, F_cdf=F_cdf_beta, n=n):
    """
    Beregner b(v) = v - ( ∫₀ᵛ [F_cdf(x)]^(n-1) dx ) / [F_cdf(v)]^(n-1)

    Parametre:
        v (float): Det punkt, hvor funktionen evalueres. Skal være 0 <= v <= 1.

    Returnerer:
        float: Værdien af b(v).
    """

    # Numerator: integral af F_cdf(x)^(n-1) fra x=0 til x=v
    numer = quad(lambda x: F_cdf(x)**(n-1), 0, v)[0]

    # Denominator: F_cdf(v)^(n-1)
    denom = F_cdf(v)**(n-1)

    return v - numer/denom
