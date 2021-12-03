
import math as m
import random as r

from vector import Vec4
from particle import Particle

class Rambo:

    def __init__(self,nin,nout,ecms):
        self.nin = nin
        self.nout = nout
        self.ecms = ecms
        pi2log = m.log(m.pi/2.)
        Z = [ 0, 0, pi2log ]
        for k in range(3,nout+1):
            Z.append(Z[k-1]+pi2log-2.*m.log(k-2))
        for k in range(3,nout+1):
            Z[k] = Z[k]-m.log(k-1)
        self.Z_N = Z[nout]

    def GenerateWeight(self,e):
        sum = Vec4()
        for i in range(self.nin):
            sum += e[i].mom
        w = m.exp((2.*self.nout-4.)*m.log(sum.M())+self.Z_N)
        w /= m.pow(2.*m.pi,self.nout*3.-4.)
        return w

    def GeneratePoint(self,e=[]):
        if len(e)==0:
            pa = Vec4(self.ecms/2,0,0,self.ecms/2)
            pb = Vec4(self.ecms/2,0,0,-self.ecms/2)
            e = [ Particle(21,-pa,[0,0]), Particle(21,-pb,[0,0]) ]
            for i in range(self.nout):
                e.append(Particle(21,Vec4(),[0,0]))
            self.GeneratePoint(e)
            return e
        sum = Vec4()
        for i in range(self.nin):
            sum += e[i].mom
        T = sum.M()
        R = Vec4()
        for i in range(self.nin,self.nin+self.nout):
            C = 2*r.random()-1.
            S = m.sqrt(1-C*C)
            F = 2*m.pi*r.random()
            Q = -m.log(min(1.-1.e-16,max(1.e-16,r.random()*r.random())))
            e[i].mom = Vec4(Q,Q*S*m.sin(F),Q*S*m.cos(F),Q*C)
            R += e[i].mom
        M = R.M()
        B = -Vec4(0.,R[1],R[2],R[3])/M
        G = R[0]/M
        A = 1./(1.+G)
        X = T/M
        for i in range(self.nin,self.nin+self.nout):
            E = e[i].mom[0]
            BQ = -B*e[i].mom
            C = e[i].mom+B*(E+A*BQ)
            e[i].mom = X*Vec4(G*E+BQ,C[1],C[2],C[3]);

if __name__== "__main__":
    # generate test point
    import argparse,time
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--nevts',default=10000,type=int,help='number of events to generate')
    parser.add_argument('-i','--nin',default=2,type=int,help='number of incoming particles')
    parser.add_argument('-o','--nout',default=3,type=int,help='number of outgoing particles')
    parser.add_argument('-e','--ecm',default=100,type=int,help='center of mass energy')
    parser.add_argument('-s','--rseed',default=123456,type=int,help='random number seed')
    args = parser.parse_args()

    from particle import CheckEvent

    rambo = Rambo(args.nin,args.nout,args.ecm)

    r.seed(args.rseed)
    start = time.time()
    for i in range(args.nevts):
        e = rambo.GeneratePoint()
    end = time.time()
    print('event rate: %d events per second' % int(args.nevts / (end - start)))