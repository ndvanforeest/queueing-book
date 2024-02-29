import matplotlib.pyplot as plt
from latex_figures import fig_in_latex_format

labda = 3
# per hour
ES0 = 15.0 / 60
# hour
ER = 2.0
x = list(range(25, 50))
y = []

for B in x:
    ESe = ES0 + ER / B
    rho = labda * ESe
    # The time to form a red batch
    labda_r = 0.5
    EW_r = (B - 1) / (2 * labda_r)
    # Now the time a batch spends in queue
    Cae = 1.0
    CaB = Cae / B
    Ce = 1.0
    # SCV of service times
    VS0 = Ce * ES0 * ES0
    VR = 1.0 * 1.0
    # Var setups is sigma squared
    VSe = B * VS0 + VR
    ESb = B * ES0 + ER
    CeB = VSe / (ESb * ESb)
    EW = (CaB + CeB) / 2 * rho / (1 - rho) * ESb
    # The time to unpack the batch, i.e., the time at the server.
    ES = ER + (B - 1) / 2 * ES0 + ES0
    total = EW_r + EW + ES
    y.append(total)


def cm_to_inch(cm):
    return cm / 2.54


plt.figure(figsize=(cm_to_inch(5), cm_to_inch(6)))
plt.xlim(20, 50)
plt.plot(x, y, label=r"$\mathsf{E}[J_r]$", lw=0.7)
plt.xlabel("$B$")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../figures/setups.pdf")
