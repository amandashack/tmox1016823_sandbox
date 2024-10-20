import psana
import numpy as np
import sys
import re
import h5py
import os
import socket

from typing import Type, List

from Fzp import *

print("Hello from Fzp_test.py")

runhsd = True
runpiranha = True
fzps = []
runnums = [22]
expname = 'tmoc00123'

port = {}
chankeys = {}
hsds = {}
hsdstring = {}

piranhas = {}

ds = psana.DataSource(exp=expname, run=runnums)
detslist = {}
hsdnames = {}
piranhanames = {}

for r in runnums:
    run = next(ds.runs())
    rkey = run.runnum

    port.update({rkey: {}})
    hsds.update({rkey: {}})
    chankeys.update({rkey: {}})
    detslist.update({rkey: [s for s in run.detnames]})

    print(port)
    print(chankeys)
    print(hsds)
    print(detslist)

    # outnames.update({rkey:'%s/hits.%s.run_%03i.h5'%(scratchdir,expname,rkey)})

    hsdslist = [s for s in detslist[rkey] if re.search('hsd', s)]  # masks for hsds
    piranhaschool = [s for s in detslist[rkey] if re.search('piranha', s)]  # masks for piranha types

    hsdnames.update({rkey: hsdslist})
    piranhanames.update({rkey: piranhaschool})

    print("piranhanames: ", piranhanames)

    # print('writing to %s'%outnames[rkey])
    for hsdname in hsdnames[rkey]:
        print(hsdname)
        port[rkey].update({hsdname: {}})
        chankeys[rkey].update({hsdname: {}})

        print(port)
        print(chankeys)
        if runhsd and hsdname in detslist[rkey]:
            hsds[rkey].update({hsdname: run.Detector(hsdname)})
            port[rkey].update({hsdname: {}})
            chankeys[rkey].update({hsdname: {}})

            print(hsds)
            print(port)
            print(chankeys)
        #     for i,k in enumerate(list(hsds[rkey][hsdname].raw._seg_configs().keys())):
        #         chankeys[rkey][hsdname].update({k:k}) # this we may want to replace with the PCIe address id or the HSD serial number.
        #         #print(k,chankeys[rkey][hsdname][k])
        #         #port[rkey][hsdname].update({k:Port(k,chankeys[rkey][hsdname][k],t0=t0s[i],logicthresh=logicthresh[i],inflate=inflate,expand=nr_expand)})
        #         port[rkey][hsdname].update({k:Port(k,chankeys[rkey][hsdname][k],inflate=inflate,expand=nr_expand)})
        #         port[rkey][hsdname][k].set_runkey(rkey).set_hsdname(hsdname)
        #         if is_fex:
        #             port[rkey][hsdname][k].setRollOn((3*int(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.xpre))>>2) # guessing that 3/4 of the pre and post extension for threshold crossing in fex is a good range for the roll on and off of the signal
        #             port[rkey][hsdname][k].setRollOff((3*int(hsds[rkey][hsdname].raw._seg_configs()[k].config.user.fex.xpost))>>2)
        #         else:
        #             port[rkey][hsdname][k].setRollOn(1<<6)
        #             port[rkey][hsdname][k].setRollOff(1<<6)
        # else:
        #     runhsd = False

    for piranhaname in piranhanames[rkey]:
        print(piranhaname)
        if runpiranha and piranhaname in detslist[rkey]:
            piranhas.update({piranhaname: run.Detector(piranhaname)})

        print(piranhas)