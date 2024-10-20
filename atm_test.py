import psana
import numpy as np
import sys
import re
import h5py
import os
import socket

from typing import Type, List

from atm import *

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
    runkey = run.runnum

    piranhas.update({runkey: {}})
    port.update({runkey: {}})
    hsds.update({runkey: {}})
    chankeys.update({runkey: {}})
    detslist.update({runkey: [s for s in run.detnames]})

    print(chankeys)
    print(hsds)
    print(detslist)

    # outnames.update({runkey:'%s/hits.%s.run_%03i.h5'%(scratchdir,expname,runkey)})

    hsdslist = [s for s in detslist[runkey] if re.search('hsd', s)]  # masks for hsds
    piranhaschool = [s for s in detslist[runkey] if re.search('piranha', s)]  # masks for piranha types

    hsdnames.update({runkey: hsdslist})
    piranhanames.update({runkey: piranhaschool})

    print("piranhanames: ", piranhanames)

    # print('writing to %s'%outnames[runkey])
    for hsdname in hsdnames[runkey]:
        print(hsdname)
        port[runkey].update({hsdname: {}})
        chankeys[runkey].update({hsdname: {}})

        print(port)
        print(chankeys)
        if runhsd and hsdname in detslist[runkey]:
            hsds[runkey].update({hsdname: run.Detector(hsdname)})
            port[runkey].update({hsdname: {}})
            chankeys[runkey].update({hsdname: {}})

            print(hsds)
            print(port)
            print(chankeys)
        #     for i,k in enumerate(list(hsds[runkey][hsdname].raw._seg_configs().keys())):
        #         chankeys[runkey][hsdname].update({k:k}) # this we may want to replace with the PCIe address id or the HSD serial number.
        #         #print(k,chankeys[runkey][hsdname][k])
        #         #port[runkey][hsdname].update({k:Port(k,chankeys[runkey][hsdname][k],t0=t0s[i],logicthresh=logicthresh[i],inflate=inflate,expand=nr_expand)})
        #         port[runkey][hsdname].update({k:Port(k,chankeys[runkey][hsdname][k],inflate=inflate,expand=nr_expand)})
        #         port[runkey][hsdname][k].set_runkey(runkey).set_hsdname(hsdname)
        #         if is_fex:
        #             port[runkey][hsdname][k].setRollOn((3*int(hsds[runkey][hsdname].raw._seg_configs()[k].config.user.fex.xpre))>>2) # guessing that 3/4 of the pre and post extension for threshold crossing in fex is a good range for the roll on and off of the signal
        #             port[runkey][hsdname][k].setRollOff((3*int(hsds[runkey][hsdname].raw._seg_configs()[k].config.user.fex.xpost))>>2)
        #         else:
        #             port[runkey][hsdname][k].setRollOn(1<<6)
        #             port[runkey][hsdname][k].setRollOff(1<<6)
        # else:
        #     runhsd = False

    for piranhaname in piranhanames[runkey]:
        print(piranhaname)
        if runpiranha: # this is redundant since piranhanames was made based off of detslist and piranhaname in detslist[runkey]:
            piranhas[runkey][piranhaname] = {'detector': run.Detector(piranhaname), 'cls': Atm(200)}
        print(piranhas)

    eventnum: int = 0
    for evt in run.events():
        completeEvent: List[bool] = [True]
        if eventnum % 10 == 0: print(eventnum)
        # run tests
        for i, pnames in enumerate(piranhas[runkey]):
            if (piranhas[runkey][pnames]['detector'] != None):
                if (type(piranhas[runkey][pnames]['detector'].raw.raw(evt)) != None):
                    completeEvent += [piranhas[runkey][pnames]['cls'].test(piranhas[runkey][pnames]['detector'].raw.raw(evt))]
                    print(completeEvent)
