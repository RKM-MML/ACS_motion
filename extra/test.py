import SPiiPlusPython as sp
point=1080
hc = sp.OpenCommEthernetTCP("192.168.0.40", 701)
#sp.Disable(hc, 0, sp.SYNCHRONOUS, True)
#sp.Enable(hc, 0, sp.SYNCHRONOUS, True)
sp.SetVelocity(hc, 0, 500, sp.SYNCHRONOUS, True)
sp.SetAcceleration(hc, 0, 9, sp.SYNCHRONOUS, True)
sp.SetDeceleration(hc, 0, 90, sp.SYNCHRONOUS, True)
sp.SetJerk(hc, 0, 100, sp.SYNCHRONOUS, True)
sp.SetKillDeceleration(hc, 0, 1000, sp.SYNCHRONOUS, True)

# sp.SetFPosition(
# hc, # communication handle
# sp.Axis.ACSC_AXIS_0, # axis 0
# 0, # required feedback position
# failure_check=True
# )

# # Initialize position
# a = 30  #steupper bound 
# b=200 #lower bound
# step=10 #step size
# # Start multi-point motion sequence
# sp.MultiPoint(
#     hc,  # Communication handle
#     0,   # Create the multi-point motion with default velocity
#     sp.Axis.ACSC_AXIS_0,
#     1000,  # Dwell time in ms
#     failure_check=True
# )
# sp.AddPoint(hc, sp.Axis.ACSC_AXIS_0, a)
# # Add points from 0 to 200 in steps of 10
# while a <= b:
#     sp.AddPoint(hc, sp.Axis.ACSC_AXIS_0, a)
#     FPOS = sp.GetFPosition(
#     hc, # communication handle
#     sp.Axis.ACSC_AXIS_0, # axis 0
#     failure_check=True
#     )
#     print(FPOS)
#     a += step

# # Finish the motion sequence
# sp.EndSequence(hc, sp.Axis.ACSC_AXIS_0)





