import scipy.constants as const

# masses in a.u.
M_Li = 7.01600342665 * 1822.888486209
m_p = const.m_p/const.m_e

# reduced proton mass
m = m_p*M_Li/(m_p + M_Li)

print(f"m={m} a.u.")
