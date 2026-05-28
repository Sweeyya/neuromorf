# Lava SDK skill for neuromorf

When building the Lava backend, always use these patterns:

## Creating a LIF Process
from lava.proc.lif.process import LIF
lif = LIF(shape=(n,), du=0, dv=0, vth=v_threshold, bias_mant=0)

## Creating Dense connections
from lava.proc.dense.process import Dense
dense = Dense(weights=weight_matrix)

## Wiring processes
lif_src.out_ports.s_out.connect(dense.in_ports.s_in)
dense.out_ports.a_out.connect(lif_dst.in_ports.a_in)

## Running
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
lif.run(condition=RunSteps(num_steps=100), run_cfg=Loihi2SimCfg())

## ALWAYS flag uncertainty about Lava API calls
## NEVER guess parameter names — check official docs