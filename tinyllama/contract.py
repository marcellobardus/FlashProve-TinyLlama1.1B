"""Constants of the TL-INT-1 arithmetic contract."""

L = 22
D = 2048
H = 32
G = 4
DH = 64
F = 5632
V = 32000
NE = 0
T = 1024
LT = 10
CS = 12
THETA0 = 10000
EXP_S = 12
S = EXP_S
RX = 8
EG = 8
SMAX = 4095
CW = 12
WEIGHT_BOUND = 4095
EMB_BOUND = 2**21
U0 = 2307
C0_BITS = 0x3FB504F3
CQ_BITS = 0x3E3504F3
PAD_ID = 0
GCL = 1
CONTRACT_ID = "TL-INT-1"

DQ = 2048
DKV = 256

RANGE_CONDITIONS = {
    1: ("Vx+eps", 0, 2**78 - 1),
    2: ("norm quotient w", -32768, 32639),
    3: ("qw, kw", -32768, 32639),
    4: ("vw", -32768, 32639),
    6: ("gh", -16384, 16383),
    7: ("u", -32768, 32639),
    8: ("stream x read by a norm", -547599908735, 547599908735),
}
