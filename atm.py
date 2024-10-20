import numpy as np


def getCentroid(data,pct=.8):
    csum = np.cumsum(data.astype(float))
    s = float(csum[-1])*pct
    csum /= csum[-1]
    inds = np.where((csum>(.5-pct/2.))*(csum<(.5+pct/2.)))
    tmp = np.zeros(data.shape,dtype=float)
    tmp[inds] = data[inds].astype(float)
    num = np.sum(tmp*np.arange(data.shape[0],dtype=float))
    return (num/s,np.uint64(s))


class Atm:
    def __init__(self, thresh) -> None:
        self.values = []
        self.vsize = int(0)
        self.vc = []
        self.vs = []
        self.initState = True
        self.atmthresh = thresh
        self.winstart = 0
        self.winstop = 1<<11
        return

    @classmethod
    def slim_update_h5(cls,f,spect,atmEvents):
        grpatm = None
        if 'atm' in f.keys():
            grpatm = f['atm']
        else:
            grpatm = f.create_group('atm')

        grpatm.create_dataset('centroids',data=spect.vc,dtype=np.float16)
        grpatm.create_dataset('sum',data=spect.vs,dtype=np.uint64)
        grpatm.attrs.create('size',data=spect.vsize,dtype=np.int32)
        grpatm.create_dataset('events',data=atmEvents)
        return

    @classmethod
    def update_h5(cls,f,spect,atmEvents):
        grpatm = None
        if 'atm' in f.keys():
            grpatm = f['atm']
        else:
            grpatm = f.create_group('atm')

        grpatm.create_dataset('data',data=spect.v,dtype=int)
        grpatm.create_dataset('centroids',data=spect.vc,dtype=np.float16)
        grpatm.create_dataset('sum',data=spect.vs,dtype=np.uint64)
        grpatm.attrs.create('size',data=spect.vsize,dtype=np.int32)
        grpatm.create_dataset('events',data=atmEvents)
        return

    def setthresh(self,x):
        self.atmthresh = x
        return self

    def setwin(self, low, high):
        self.winstart = int(low)
        self.winstop = int(high)
        return self

    def test(self,atmwv):
        mean = np.int16(0)
        if type(atmwv)==type(None):
            return False
        try:
            mean = np.int16(np.mean(atmwv[800:])) # this subtracts baseline
        except:
            print('Damnit, atm!')
            return False
        else:
            if (np.max(atmwv)-mean)<self.atmthresh:
                #print('weak atm!')
                return False
        return True

    def process(self, atmwv):
        mean = np.int16(np.mean(atmwv[1800:])) # this subtracts baseline
        if (np.max(atmwv)-mean)<self.atmthresh:
            return False
        d = np.copy(atmwv-mean).astype(np.int16)
        c, s = getCentroid(d[self.winstart:self.winstop],pct=0.8)
        if self.initState:
            self.v = [d]
            self.vsize = len(self.v)
            self.vc = [np.float16(c)]
            self.vs = [np.uint64(s)]
            self.initState = False
        else:
            self.v += [d]
            self.vc += [np.float16(c)]
            self.vs += [np.uint64(s)]
        return True

    def set_initState(self,state: bool):
        self.initState = state
        return self

    def print_v(self):
        print(self.v[:10])
        return self