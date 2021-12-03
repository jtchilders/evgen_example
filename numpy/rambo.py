#!/usr/bin/env python
# a port of the pythonic version to numpy vectorized
import numpy as np
import logging
logger = logging.getLogger(__name__)


def vectmultiply(a,b):
    c = a * b
    return c[...,0] - c[...,1] - c[...,2] - c[...,3]


class Rambo:

    def __init__(self,nevts,nin,nout,ecms):
        logger.debug('init: nevts = %s; nin = %s; nout = %s; ecms = %s',nevts,nin,nout,ecms)
        self.nevts = nevts
        self.nin = nin
        assert nin == 2
        self.nout = nout
        self.ecms = ecms
        pi2log = np.log(np.pi/2.)
        Z = [ 0, 0, pi2log ]
        for k in range(nin+1,nout+1):
            Z.append(Z[k-1]+pi2log-2.*np.log(k-2))
        for k in range(nin,nout+1):
            Z[k] = Z[k]-np.log(k-1)
        self.Z_N = Z[nout]

    def get_inputs(self):
        # input_particles = np.zeros([self.nevts,self.nin,4])

        pa = np.array([self.ecms / 2.,0.,0.,self.ecms / 2])
        pb = np.array([self.ecms / 2.,0.,0.,-self.ecms / 2])

        input_particles = np.array([pa,pb])
        input_particles = np.repeat(input_particles[np.newaxis,...],self.nevts,axis=0)
        logger.debug('input_particles = %s',input_particles)

        return input_particles

    @staticmethod
    def get_momentum_sum(inarray):
        return np.sum(inarray,axis=1)

    @staticmethod
    def get_combined_mass(inarray):
        sum  = Rambo.get_momentum_sum(inarray)
        logger.debug('input_sum = %s',sum)
        return Rambo.get_mass(sum)

    @staticmethod
    def get_mass(inarray):
        mom2 = np.sum(inarray[...,1:4]**2,axis=1)
        mass = np.sqrt(inarray[...,0]**2 - mom2)
        return mass

    def get_output_mom2(self):

        C1 = np.zeros([self.nevts,self.nout])
        F1 = np.zeros([self.nevts,self.nout])
        Q1 = np.zeros([self.nevts,self.nout])

        for i in range(self.nevts):
            for j in range(self.nout):
                C1[i,j] = np.random.rand()
                F1[i,j] = np.random.rand()
                Q1[i,j] = np.random.rand()*np.random.rand()
                logger.debug('i,j = %s,%s; C1 = %s; F1 = %s; Q1 = %s',i,j,C1[i,j],F1[i,j],Q1[i,j])


        C = 2.*C1-1.
        logger.debug('C = %s',C)
        S = np.sqrt(1 - C**2)
        logger.debug('S = %s',S)
        F = 2.*np.pi*F1
        logger.debug('F = %s',F)
        Q = -np.log(Q1)
        logger.debug('Q = %s',Q)
        output = np.zeros([self.nevts,self.nout,4])
        output[...,0] = Q
        output[...,1] = Q*S*np.sin(F)
        output[...,2] = Q*S*np.cos(F)
        output[...,3] = Q*C

        return output

    def get_output_mom(self):
        C = 2.*np.random.rand(self.nevts,self.nout)-1.
        logger.debug('C = %s',C)
        S = np.sqrt(1 - C**2)
        logger.debug('S = %s',S)
        F = 2.*np.pi*np.random.rand(self.nevts,self.nout)
        logger.debug('F = %s',F)
        Q = -np.log(np.random.rand(self.nevts,self.nout)*np.random.rand(self.nevts,self.nout))
        logger.debug('Q = %s',Q)
        output = np.zeros([self.nevts,self.nout,4])
        output[...,0] = Q
        output[...,1] = Q*S*np.sin(F)
        output[...,2] = Q*S*np.cos(F)
        output[...,3] = Q*C

        return output


    def GeneratePoints(self):
        input_particles = self.get_inputs()

        input_mass = self.get_combined_mass(input_particles)

        logger.debug('input_mass = %s',input_mass)

        output_particles = self.get_output_mom2()
        logger.debug('output_particles = %s',output_particles)

        output_mom_sum = self.get_momentum_sum(output_particles)
        logger.debug('output_mom_sum = %s',output_mom_sum)
        output_mass = self.get_mass(output_mom_sum)
        logger.debug('ouput_mass = %s',output_mass)

        G = output_mom_sum[...,0] / output_mass
        G = np.repeat(G[...,np.newaxis],self.nout,axis=1)
        logger.debug('G = %s',G)
        X = input_mass / output_mass
        X = np.repeat(X[...,np.newaxis],self.nout,axis=1)
        logger.debug('X = %s',X)

        output_mass = np.repeat(output_mass[...,np.newaxis],3,axis=1)
        logger.debug('output_mass = %s',output_mass)

        B = np.zeros(output_mom_sum.shape)
        B[...,1:4] = -output_mom_sum[...,1:4] / output_mass
        B = np.repeat(B[:,np.newaxis,:],self.nout,axis=1)
        logger.debug('B = %s',B)

        A = 1. / (1. + G)
        logger.debug('A = %s',A)

        E  = output_particles[...,0]
        logger.debug('E = %s',E)
        BQ = -1. * vectmultiply(B,output_particles)
        logger.debug('BQ = %s',BQ)
        C1 = E + A * BQ
        logger.debug('C1 = %s',C1)
        C1 = np.repeat(C1[...,np.newaxis],4,axis=2)
        logger.debug('C1 = %s',C1)
        C = output_particles + B * C1
        logger.debug('C = %s',C)
        D = G * E + BQ
        logger.debug('D = %s',D)
        output_particles[...,0] = X * D
        output_particles[...,1:4] = np.repeat(X[...,np.newaxis],3,axis=2) * C[...,1:4]
        logger.debug('output_particles = %s',output_particles)

        return np.concatenate((input_particles,output_particles),axis=1)
        


if __name__ == "__main__":
    # generate test
    logging.basicConfig(level=logging.INFO)

    # from particle import CheckEvent

    import argparse,time
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--nevts',default=10000,type=int,help='number of events to generate')
    parser.add_argument('-v','--vevts',default=100,type=int,help='number of events stored per vector')
    parser.add_argument('-i','--nin',default=2,type=int,help='number of incoming particles')
    parser.add_argument('-o','--nout',default=3,type=int,help='number of outgoing particles')
    parser.add_argument('-e','--ecm',default=100,type=int,help='center of mass energy')
    parser.add_argument('-s','--rseed',default=123456,type=int,help='random number seed')
    args = parser.parse_args()

    rambo = Rambo(args.vevts,args.nin,args.nout,args.ecm)
    np.random.seed(args.rseed)

    
    nruns = int(args.nevts / args.vevts) + 1
    start = time.time()
    for i in range(nruns):
        e  = rambo.GeneratePoints()
    end = time.time()
    print('event rate: %d events per second' % int(args.nevts / (end - start)))



