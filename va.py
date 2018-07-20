#! /usr/bin/env python3
# -*- coding: utf8 -*-

##########################################################################
#                                                                        #
# Program: va.py   Version 1.2              Date: 05/07/2018             #
#                                                                        #
# A varied set of tools to analyse vibrational modes in terms of         #
# localized internal coordinates.                                        #
#                                                                        #
# (c) Filipe Teixeira, 2017                                              #
#                                                                        #
##########################################################################

import numpy as np
import sys
import sklearn.linear_model as sklm
import sklearn.metrics as skmt

# General Options
Opts={}
Opts['isLinear']=False
Opts['isTState']=False
Opts['cut']='auto'
Opts['cutval']=0.1
Opts['delta']=0.01
Opts['tol']=0.2
Opts['doOuts']=True
Opts['doTors']=True
Opts['doMWD']=True
Opts['doMWS']=False
Opts['doAutoSel']=False
Opts['doVMP']=False
Opts['doVMLD']=False
Opts['doVMBLD']=False
Opts['doVMARD']=True
Opts['input']='OrcaHess'
Opts['aniMode']=[]
Opts['anic']=[]
Opts['oopp']=True
Opts['oopTol']=np.deg2rad(5.0)

Linear=False
Transition=False

# General Data
Bohr2Ang=0.529

Symbols=['h','he','li','be','b','c','n','o','f','ne',
'na','mg','al','si','p','s','cl','ar', 'k','ca','sc',
'ti','v','cr','mn','fe','co','ni','cu','zn','ga','ge',
'as','se','br','kr', 'rb', 'sr', 'y', 'zr', 'nb', 'mo', 
'tc','ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te',
'i', 'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 
'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu',
'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi']

Masses=[1.007825, 4.002602, 6.94, 9.0121831, 10.81, 12.0000, 14.007, 15.9949159, 18.998403163, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085, 30.973761998, 32.06, 35.45, 39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.95, 97, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.90545196, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804]

Raddi=[ 0.5, 0.3, 1.7, 1.1, 0.9, 0.7, 0.6, 0.6, 0.5, 0.4, 1.9, 1.5, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 2.4, 1.9, 1.8, 1.8, 1.7, 1.7, 1.6, 1.6, 1.5, 1.5, 1.5, 1.4, 1.4, 1.3, 1.1, 1.0, 0.9, 0.9, 2.7, 2.2, 2.1, 2.1, 2.0, 1.9, 1.8, 1.8, 1.7, 1.7, 1.7, 1.6, 1.6, 1.5, 1.3, 1.2, 1.2, 1.1, 3.0, 2.5, 2.0, 2.0, 2.5, 2.1, 2.1, 2.4, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.2, 2.2, 2.2, 2.1, 2.0, 1.9, 1.9, 1.9, 1.8, 1.8, 1.7, 1.7, 1.6, 1.5, 1.4]

#balancing parameters
sangles=1.0
souts=1.0
storsions=1.0

## Object Classes ##

class Atom():
	def __init__(self,symbol,r,mass=None):
		self.symbol=symbol.capitalize()
		self.mass=mass
		if(mass==None):
			mass=Masses[Symbols.index(symbol.lower())]
		self.r=np.array(r)

class Vibration():
	def __init__(self,freq,ndegs,adv=[],ir=None,raman=None,sym=''):
		self.frequency=float(freq)
		if(len(adv)==0):
			self.displacements=np.zeros(ndegs)
		else:
			self.displacements=np.array(adv)
		self.intIR=ir
		self.intRaman=raman
		# these are just the coefficients, make
		# sure the list of ICs does not change during
		# the run of the program !!!
		self.analysis={}
		self.VMP=None
		self.VMLD=None
		self.VMARD=None
		self.symmetry=sym
	def editDisplacements(self,d):
		if(len(d)!=len(self.displacements)):
			print("Warnning: New displacements do not conform to previous.")
		self.displacements=np.array(d)
	def editIntensity(self,val,t='IR'):
		if(t.upper()=='IR'):
			self.intIR=float(val)
		elif(t.upper()=='RAMAN'):
			self.intRaman=float(val)
	def addVMP(self,vmp):
		self.analysis['VMP']=np.array(vmp)
	def addVMLD(self,coefs,r2=0.0,exvar=0.0):
		self.analysis['VMLD']=np.array(coefs)
		self.analysis['VMLD_R2']=r2
		self.analysis['VMLD_EV']=exvar
	def addVMBLD(self,coefs,r2=0.0,exvar=0.0):
		self.analysis['VMBLD']=np.array(coefs)
		self.analysis['VMBLD_R2']=r2
		self.analysis['VMBLD_EV']=exvar
	def addVMARD(self,coefs,r2=0.0,exvar=0.0):
		self.analysis['VMARD']=np.array(coefs)
		self.analysis['VMARD_R2']=r2
		self.analysis['VMARD_EV']=exvar
	def string(self,n=-1):
		o=''
		if(n>0):
			o="Mode %3d: %8.2f cm-1 "%(n,self.frequency)
		else:
			o="Mode at %8.2f cm-1 "%(self.frequency)
		if(self.symmetry.strip()!=''):
			o += "(%s) "%(self.symmetry)
		if(self.intIR != None):
			o += "(IR: %5.1f"%(self.intIR)
		if((self.intIR == None)and(self.intRaman != None)):
			o += "(Raman: %5.1f)"%(self.intRaman)
		if((self.intIR != None)and(self.intRaman != None)):
			o += ",Raman: %5.1f)"%(self.intRaman)
		if(self.intRaman == None):
			o += ")"
		return o

class System():
	def __init__(self):
		self.atoms=[]
		self.geo=np.array([])
		self.natoms=0
		self.symbol=[]
		self.vibrations=[]
		self.intcoords=[]
		self.S=np.array([])
		self.ADM=None
	def addAtom(self,symbol,pos,mass=None):
		self.atoms.append(Atom(symbol,pos,mass))
		self.natoms = len(self.atoms)
	def addVibration(self,frequency):
		self.vibrations.append(Vibration(frequency,3*self.natoms))
	def addVibIntensity(self,idx,val,kind='IR'):
		self.vibrations[idx].editIntensity(val,kind)
	def addVibSymmetry(self,idx,sym=''):
		self.vibrations[idx].symmetry=sym
	def addDisplacements(self,idx,displacements):
		self.vibrations[idx].editDisplacements(displacements)
	def makeADM(self):
		self.ADM=np.zeros((3*self.natoms,len(self.vibrations)))
		for i in range(len(self.vibrations)):
			self.ADM[:,i]=self.vibrations[i].displacements
	def checkADM(self):
		"""Check orthogonalyty of the ADM"""
		self.makeADM()
		for n in range(self.ADM.shape[1]):
			norm=np.linalg.norm(self.ADM[:,n])
			#print("Norm of Vibrational Mode %d is %8.6f"%(n+1,norm))
			for m in range(n+1,self.ADM.shape[1]):
				c=np.dot(self.ADM[:,n],self.ADM[:,m])
				#print("Cross between Mode %d and Mode %d is: %8.6f"%(n+1,m+1,c))
	def makeGeo(self):
		self.geo=np.zeros((self.natoms,3))
		self.symbol=[]
		for i in range(self.natoms):
			self.geo[i,:]=self.atoms[i].r
			self.symbol.append(self.atoms[i].symbol)
	def tripleMass(self):
		o=np.zeros(3*self.natoms)
		n=0
		for i in range(self.natoms):
			#print(" %d %2s %5.2f %10.6f %10.6f %10.6f"%(i+1,self.atoms[i].symbol,self.atoms[i].mass,self.atoms[i].r[0],self.atoms[i].r[1],self.atoms[i].r[2]))
			for k in range(3):
				o[n]=self.atoms[i].mass
				n += 1
		return o
	def massWeightS(self):
		mass=np.sqrt(self.tripleMass())
		for i in range(self.S.shape[1]):
			self.S[:,i] /= mass
	def normalizeS(self):
		for i in range(self.S.shape[1]):
			self.S[:,i] /= np.linalg.norm(self.S[:,i])
		#self.S *= Opts["delta"] #do we need this?
	def massWeightVibrations(self):
		mass=np.array(self.tripleMass())
		for i in range(len(self.vibrations)):
			self.vibrations[i].displacements *= mass
	def normalizeVibrations(self):
		for i in range(len(self.vibrations)):
			self.vibrations[i].displacements /= np.linalg.norm(self.vibrations[i].displacements)
	def removeVibrations(self,elst=list(range(6))):
		idx=[]
		for i in range(len(self.vibrations)):
			if(i not in elst):
				idx.append(i)
		self.vibrations=[self.vibrations[i] for i in idx]
	def sortVibrations(self):
		self.vibrations=sorted(self.vibrations,key=lambda x: x.frequency, reverse=True)

## Input-Output Functions ##

def readMopac2016(ifn):
	"""Opens a MOPAC2016 output file ifn and returns A System object.
	Depending on Linear and Transition, freqs and modes will be
	pruned out of the translational and rotational components."""
	f=open(ifn,'r')
	data=f.readlines()
	f.close()
	o=System()
	# read geometry (in angs) and atomic symbols, get masses from internal lib
	natoms=-1
	for i in range(len(data)):
		if('Empirical Formula:' in data[i]):
			natoms=int(data[i].split()[-2])
			break
	if(natoms<1):
		print('ERROR: Could not read MOPAC output: no empirical formula!\n')
		sys.exit(1)
	for i in range(len(data)):
		if ('ORIENTATION OF MOLECULE IN FORCE CALCULATION' in data[i]):
			for j in range(i+4,i+4+natoms):
				l=data[j].split()
				o.addAtom(l[1].lower(),np.array(list(map(float,l[2:5]))),Masses[Symbols.index(l[1].lower())])
			break
	# read frequency list and displacements
	istart=-1
	iend=-1
	for i in range(len(data)):
		if(' NORMAL COORDINATE ANALYSIS' in data[i]):
			istart=i+1
		if('MASS-WEIGHTED COORDINATE ANALYSIS' in data[i]):
			iend=i-1
			break
	if((istart<0) or(iend<istart)):
		print('ERROR: Could not read MOPAC output: invalid format for normal coordinate analysis.\n')
		sys.exit(1)
	tfreq=[]
	tir=[]
	tsym=[]
	tdisp=[]
	n=istart
	curr=0
	last=0
	while(n<iend):
		if('Root No.' in data[n]):
			curr=len(data[n].split())-2
			for i in range(curr):
				tdisp.append([])
			#reading symmetries
			n += 2
			l=data[n].split()
			for i in range(1,len(l),2):
				tsym.append(l[i])
			#reading frequencies
			n += 2
			l=data[n].split()
			for i in l:
				tfreq.append(float(i))
			#reading displacements
			n += 1
			for i in range(3*natoms):
				n += 1
				l=data[n].split()[1:]
				for j in range(last,last+curr):
					tdisp[j].append(l[j-last])
			last += curr
		n += 1
	# read T-DIPOLE as sorrrugate for IR intensities
	for i in range(len(data)):
		if('T-DIPOLE' in data[i]):
			tir.append(float(data[i].split()[1]))
	# assemble output System
	putir=True
	if(len(tir)!=len(tfreq)):
		print("Warnning: Could not read IR intensities.")
		putir=False
	for i  in range(len(tfreq)):
		if(putir):
			o.vibrations.append(Vibration(tfreq[i],3*natoms,np.array(list(map(float,tdisp[i]))),ir=tir[i],sym=tsym[i]))
		else:
			o.vibrations.append(Vibration(tfreq[i],3*natoms,np.array(list(map(float,tdisp[i]))),sym=tsym[i]))
	return o


def readHess(ifn):
	"""Opens Orca Hess file ifn and returns A System object.
	Depending on Linear and Transition, freqs and modes will be
	pruned out of the translational and rotational components."""
	f=open(ifn,'r')
	data=f.readlines()
	f.close()
	o=System()
	# read masses, geometry (in angs) and atomic symbols
	for i in range(len(data)):
		if ('$atoms' in data[i]):
			natoms=int(data[i+1])
			for j in range(i+2,i+2+natoms):
				l=data[j].split()
				o.addAtom(l[0].lower(),np.array(list(map(float,l[2:5])))*Bohr2Ang,float(l[1]))
			break
	# read frequency list
	for i in range(len(data)):
		if ('$vibrational_frequencies' in data[i]):
			nfreqs=int(data[i+1])
			for j in range(i+2,i+2+nfreqs):
				o.addVibration(float(data[j].split()[1]))
			break
	# read normal mode displacements
	for i in range(len(data)):
		if ('$normal_modes' in data [i]):
			ndegs=int(data[i+1].split()[0])
			ipos=i+2
			break
	modes=np.zeros((ndegs,ndegs))
	#count number of cols on first line
	incr=len(data[ipos+1].split())-1
	nils=int(ndegs/incr)
	nrem=np.mod(ndegs,incr)
	#epos=ipos+((ndegs+1)*nils)+1 #old versions of Orca?
	epos=ipos+((ndegs+1)*nils)
	if (nrem>0): epos+= (ndegs+1)
	icol=0
	ecol=0
	nread=0
	for n in range(ipos,epos,ndegs+1):
		icol=ecol
		ecol=icol+incr
		nread+=1
		if (nread>nils): ecol=ndegs+1
		l=0
		for i in range(n+1,n+ndegs+1):
			line=data[i].split()[1:]
			modes[l,icol:ecol]=list(map(float,line))
			l+=1
	for i in range(ndegs):
		o.addDisplacements(i,modes[:,i])
	# read IR intensities
	found=False
	for i in range(len(data)):
		if ('$ir_spectrum' in data [i]):
			nfreqs=int(data[i+1])
			for j in range(nfreqs):
				o.addVibIntensity(j,float(data[j+i+2].split()[1]),'IR')
			found=True
			break
	if(not found):
		print("Warnning: IR intensities not found.")
	# read Raman intensities
	found=False
	for i in range(len(data)):
		if ('$raman_spectrum' in data [i]):
			nfreqs=int(data[i+1])
			for j in range(nfreqs):
				o.addVibIntensity(j,float(data[j+i+2].split()[1]),'Raman')
			found=True
			break
	if(not found):
		print("Warnning: Raman intensities not found.")
	# prune rotational and translational modes from freqs, modes, and Ints
	if(Opts['isLinear'] and Opts['isTState']):
		excl=list(range(1,6))
	elif(Opts['isLinear'] and not Opts['isTState']):
		excl=list(range(5))
	elif((not Opts['isLinear']) and Opts['isTState']):
		excl=list(range(1,7))
	else:
		excl=list(range(6))
	o.removeVibrations(excl)
	return o

def readG09log(ifn):
	"""Opens Gaussian09 log file ifn and returns a System object with
	information regarding the system and vibrations.
	Depending on Linear and Transition, freqs and modes will be
	pruned out of the translational and rotational components."""
	o=System()
	f=open(ifn,'r')
	data=f.readlines()
	f.close()
	# get number of atoms
	natoms=-1
	for i in range(len(data)):
		if('NAtoms=' in data[i]):
			natoms=int(data[i][8:14])
			break
	symbols=[]
	mass=np.zeros(natoms)
	geo=np.zeros((natoms,3))
	p=0
	#read (last) geometry in standard orientation
	istart=-1
	for i in range(len(data)):
		if('Standard orientation:' in data[i]):
			istart=i+5
	if(istart>0):
		for i in range(istart,istart+natoms):
			l=data[i].split()
			o.addAtom(Symbols[int(l[1])-1],list(map(float,l[-3:])))
	#read masses
	istart = -1
	for i in range(len(data)):
		if('- Thermochemistry -' in data[i]):
			istart=i+3
			break
	if(istart>0):
		for i in range(istart,istart+natoms):
			l=data[i].split()
			o.atoms[int(l[1])-1].mass=float(l[-1])
	else:	
		print("ERROR: Cannot read geometry!\n")
		sys.exit(1)
	# read vibrational frequencies, modes, IR and Raman Intensities
	nfreqs=0
	freqs=[]
	irInt=[]
	raInt=[]
	sym=[]
	modes=np.zeros((3*natoms,0))
	for i in range(len(data)):
		mstart=-1
		if('Frequencies -- ' in data[i]):
			# read symmetries
			l=data[i-1].split()
			for f in l:
				sym.append(f)
			l=data[i].split()[2:]
			freqsInRow=len(l)
			for f in l:
				freqs.append(float(f))
			for j in range(i+1,i+9):
				if('Atom' in data[j]):
					mstart=j+1
					break
				if('IR Inten    --' in data[j]):
					l=data[j].split()[3:]
					for f in l:
						irInt.append(float(f))
				if('Raman Activ --' in data[j]):
					l=data[j].split()[3:]
					for f in l:
						raInt.append(float(f))
			lmodes=[]
			for k in range(freqsInRow):
				lmodes.append([])
			for j in range(mstart,mstart+natoms):
				p=0
				l=list(map(float,data[j].split()[2:]))
				for k in range(freqsInRow):
					lmodes[k].append(l[k+p]) #dx
					p += 1
					lmodes[k].append(l[k+p]) #dy
					p += 1
					lmodes[k].append(l[k+p]) #dz
			lmodes=np.transpose(np.array(lmodes))
			modes=np.concatenate((modes,lmodes),axis=1)
	if(len(irInt)==0):
		print("Warnning: IR intensities not found.")
		irInt=np.zeros(len(freqs))
	if(len(raInt)==0):
		print("Warnning: Raman intensities not found.")
		raInt=np.zeros(len(freqs))
	# send information to System:
	for n in range(len(freqs)):
		o.vibrations.append(Vibration(freqs[n],3*natoms,modes[:,n], irInt[n],raInt[n],sym[n]))
	# normalize normal modes
	#for i in range(np.shape(modes)[1]):
	#	modes[:,i]=modes[:,i]/np.linalg.norm(modes[:,i])
	return o

def readUserIC(fn):
	"""Reads additional internal coordinates defined by the user"""
	o=[]
	f=open(fn,'r')
	udata=f.readlines()
	f.close()
	for line in udata:
		if('#' in line):
			continue
		elif(line.strip()==''):
			continue
		else:
			l=line.split()
			if(l[0].upper()=='B'):
				o.append([1,int(l[1])-1,int(l[2])-1])
			elif(l[0].upper()=='A'):
				o.append([2,int(l[1])-1,int(l[2])-1,int(l[3])-1])
			elif(l[0].upper()=='O'):
				o.append([3,int(l[1])-1,int(l[2])-1,int(l[3])-1,int(l[4])-1])
			elif(l[0].upper()=='T'):
				o.append([4,int(l[1])-1,int(l[2])-1,int(l[3])-1,int(l[4])-1])
			else:
				continue
	return(o)

def modeStr(mode,symbs):
	"""returns a formatted string describing mode"""
	o=""
	if(mode[0]==1): # bond
		o="BOND %s%d %s%d"%(symbs[mode[1]].capitalize(),mode[1]+1,symbs[mode[2]].capitalize(),mode[2]+1)
	elif(mode[0]==2): # ANGLE
		o="ANGLE %s%d %s%d %s%d"%(symbs[mode[1]].capitalize(),mode[1]+1,symbs[mode[2]].capitalize(),mode[2]+1,symbs[mode[3]].capitalize(),mode[3]+1)
	elif(mode[0]==3): # OUT OF PLANE
		o="OUT %s%d %s%d %s%d %s%d"%(symbs[mode[1]].capitalize(),mode[1]+1,symbs[mode[2]].capitalize(),mode[2]+1,symbs[mode[3]].capitalize(),mode[3]+1,symbs[mode[4]].capitalize(),mode[4]+1)
	elif(mode[0]==4): # TORSION
		o="TORSION %s%d %s%d %s%d %s%d"%(symbs[mode[1]].capitalize(),mode[1]+1,symbs[mode[2]].capitalize(),mode[2]+1,symbs[mode[3]].capitalize(),mode[3]+1,symbs[mode[4]].capitalize(),mode[4]+1)
	return(o)

def punchIC(o,s):
	"""Punches a list of internal coordinates and their
	meassured values to file o"""
	o.write("\n\nList of Internal Coordinates and their values\n")
	for i in range(len(s.intcoords)):
		desc=modeStr(s.intcoords[i],s.symbol)
		if(s.intcoords[i][0]==1): #bond
			val=bondLength(s.geo,s.intcoords[i][1:3])
		elif(s.intcoords[i][0]==2): #angle
			val=angleAmp(s.geo,s.intcoords[i][1:4],True)
		elif(s.intcoords[i][0]==3): # out
			val=oopAmp(s.geo,s.intcoords[i][1:5],True)
		elif(s.intcoords[i][0]==4): # TORS
			val=torsionAmp(s.geo,s.intcoords[i][1:5],True)
		o.write(" %4d %-25s %8.3f\n"%(i+1,desc,val))
	o.write("\n")

def animateMode(tfn,s,m,nsteps=50,damp=0.33):
	"""displaces geo over vibrational displacement m and punches a xyz file tfn"""
	natoms=len(s.atoms)
	tf=open(tfn,'w')
	v=damp*np.sin(np.linspace(0-0,2.0*np.pi,nsteps))
	for n in range(nsteps):
		g = s.geo.copy()
		g = g.reshape(g.size)+(v[n]*s.vibrations[m].displacements)
		tf.write(" %d \n Generated by vibAnalysis d=%7.4f\n"%(natoms,v[n]))
		for i in range(natoms):
			tf.write(" %3s %10.6f %10.6f %10.6f\n"%(s.symbol[i],g[3*i],g[(3*i)+1],g[(3*i)+2]))
	tf.close()

def animateIC(tfn,s,m,nsteps=50,damp=0.33):
	"""displaces geo over internal coordinate m and punches a xyz file tfn"""
	natoms=len(s.atoms)
	tf=open(tfn,'w')
	v=damp*np.sin(np.linspace(0-0,2.0*np.pi,nsteps))
	for n in range(nsteps):
		g = s.geo.copy()
		g = g.reshape(g.size)+(v[n]*s.S[:,m])
		tf.write(" %d \n Generated by vibAnalysis d=%7.4f\n"%(natoms,v[n]))
		for i in range(natoms):
			tf.write(" %3s %10.6f %10.6f %10.6f\n"%(s.symbol[i],g[3*i],g[(3*i)+1],g[(3*i)+2]))
	tf.close()

def printResults(o,s):
	"""Prints the results of the analysis stored in system s onto file o"""
	lanalysis=['VMP','VMLD','VMBLD','VMARD']
	for a in lanalysis:
		if(a in s.vibrations[0].analysis.keys()):
			if(a=='VMP'):
				o.write("\n\n*** Vibrational Mode Projection (VMP) ***\n")
			elif(a=='VMLD'):
				o.write("\n\n*** Vibrational Mode Linear Decomposition (VMLD) ***\n")
			elif(a=='VMBLD'):
				o.write("\n\n*** Vibrational Mode Bayesian Linear Decomposition (VMBLD) ***\n")
			elif(a=='VMARD'):
				o.write("\n\n*** Vibrational Mode Automatic Relevance Determination (VMARD) ***\n")
			n=1
			for vib in s.vibrations:
				o.write('\n'+vib.string(n)+'\n')
				#calculate weights
				w=vib.analysis[a]/np.sum(np.abs(vib.analysis[a]))
				n += 1
				cut = 0.0 # cut = all
				if(Opts['cut'] == 'auto'):
					cut = len(s.vibrations)/(10.0*len(s.intcoords))
				elif(Opts['cut'] == 'val'):
					cut = Opts['cutval']
				elif(Opts['cut'] == 'q1'):
					cut = np.percentile(w,75)
				elif(Opts['cut']=='d9'):
					cut = np.percentile(w,90)
				widx = list(np.argsort(np.abs(w)))
				widx.reverse()
				gw=np.zeros(5)
				gwr=np.zeros(5)
				natts=0
				for j in widx:
					gw[s.intcoords[j][0]] += np.abs(w[j])
					if(np.abs(w[j])>=cut):
						gwr[s.intcoords[j][0]] += np.abs(w[j])
						natts += 1
						of.write(" %+8.4f (%5.1f%%) %s\n"%(vib.analysis[a][j],
						np.abs(w[j])*100,modeStr(s.intcoords[j],s.symbol)))
				if(natts==0):
					of.write(" Mode too disperse!\n")
					of.write(" Largest Contribution: \n")
					j=np.argmax(w)
					of.write(" %+8.4f (%5.1f%%) %s\n"%(vib.analysis[a][j],np.abs(w[j])*100,
					modeStr(s.intcoords[j],s.symbol)))
				else:
					gwr[1:]=gwr[1:]/np.sum(gwr[1:])
				of.write(" Shown Composition:  %5.1f%% BOND, %5.1f%% ANGLE, %5.1f%% OUT, %5.1f%% TOR\n"%tuple(100.0*(gwr[1:])))
				of.write(" Total Composition:  %5.1f%% BOND, %5.1f%% ANGLE, %5.1f%% OUT, %5.1f%% TOR\n"%tuple(100.0*gw[1:]))
				if( a in ['VMLD','VMBLD','VMARD']): #punch additional info
					o.write(" R**2 = %6.4f\n"%(vib.analysis["%s_R2"%(a)]))
					o.write(" Explained Variance = %5.1f %%\n"%(vib.analysis["%s_EV"%(a)]*100.0))
			if(a=='VMP'):
				o.write("\n*** End of Vibrational Mode Projection (VMP) ***\n")
			elif(a=='VMLD'):
				o.write("\n*** End of Vibrational Mode Linear Decomposition (VMLD) ***\n")
			elif(a=='VMBLD'):
				o.write("\n*** End of Vibrational Mode Bayesian Linear Decomposition (VMBLD) ***\n")
			elif(a=='VMARD'):
				o.write("\n*** End of Vibrational Mode Automatic Relevance Determination (VMARD) ***\n")

## Support Functions ##

def bondLength(geo,bl):
	"""Returns distance between atoms bl[1] and bl[0]
	(numbering starts at 0) in geometry geo"""
	return np.linalg.norm(geo[bl[1]]-geo[bl[0]])

def angleAmp(geo,al,deg=False):
	"""Returns amplitude (optnialy in degs) for the valence angle formed by
	the atoms in al, with al[1] being the apex (numbering starts at 0)."""
	r01=geo[al[0]]-geo[al[1]]
	r21=geo[al[2]]-geo[al[1]]
	r01=r01/np.linalg.norm(r01)
	r21=r21/np.linalg.norm(r21)
	phi=np.arccos(np.dot(r01,r21))
	if(deg):
		phi=np.rad2deg(phi)
	return phi

def oopAmp(geo,al,deg=False):
	"""Returns amplitude (optnialy in degs) for the out-of-plane angle formed by
	the atoms in al, with al[1] being the central atom (numbering starts at 0)."""
	r1=geo[al[0]]-geo[al[1]]
	r2=geo[al[2]]-geo[al[1]]
	r3=geo[al[3]]-geo[al[1]]
	r1 /= np.linalg.norm(r1)
	r2 /= np.linalg.norm(r2)
	r3 /= np.linalg.norm(r3)
	n1=np.cross(r1,r2)
	n2=np.cross(r3,r2)
	atmp=np.dot(n1,n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))
	if(atmp>1.0):
		phi=np.arccos(1.0)
	elif(atmp<-1.0):
		phi=np.arccos(-1.0)
	else:
		phi=np.arccos(atmp)
	# r1 is normal to the reference plane, so
	# project n1 and n2 onto the reference plane
	n1p = n1 - (np.dot(n1,r1)*r1)
	n2p = n2 - (np.dot(n2,r1)*r1)
	nc=np.cross(n1p,n2p)
	if(np.dot(nc,r1)>0.0):
		phi = (2.0*np.pi)-phi
	if(deg):
		phi=np.rad2deg(phi)
	return phi

def torsionAmp(geo,al,deg=False):
	"""Returns amplitude (optnialy in degs) for the 0-1-2-3 torsion angle formed by
	the atoms in al, with al[1] and al[2] defining the central bond
	(numbering starts at 0)."""
	r1=geo[al[0]]-geo[al[1]]
	r2=geo[al[2]]-geo[al[1]]
	r3=geo[al[3]]-geo[al[2]]
	r1 /= np.linalg.norm(r1)
	r2 /= np.linalg.norm(r2)
	r3 /= np.linalg.norm(r3)
	n1=np.cross(r1,r2)
	n2=np.cross(r3,r2)
	if(np.linalg.norm(n1)<1.0e-3):
		# possible hazard for linear bonds!!!
		return 0.0
	elif(np.linalg.norm(n2)<1.0e-3):
		return 0.0
	n1 /= np.linalg.norm(n1)
	n2 /= np.linalg.norm(n2)
	atmp=np.dot(n1,n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))
	if(atmp>1.0):
		phi=np.arccos(1.0)
	elif(atmp<-1.0):
		phi=np.arccos(-1.0)
	else:
		phi=np.arccos(atmp)
	# r2 is normal to the reference plane, so
	# project on n1 and n2 onto the reference plane
	n1p = n1 - (np.dot(n1,r2)*r2)
	n2p = n2 - (np.dot(n2,r2)*r2)
	nc=np.cross(n1p,n2p)
	if(np.dot(nc,r1)>0.0):
		phi = (2.0*np.pi)-phi
	if(deg):
		phi=np.rad2deg(phi)
	return phi

def makeIC(o,useric=[]):
	"""Automatically identifies internal coordinates using 
	connectivity deduced from covalent radii and generates
	Wilson's S matrix for the specified geometry"""
	## Parameters from Global Opts
	delta=Opts['delta']
	tol=Opts['tol']
	o.makeGeo()
	geo=np.copy(o.geo)
	o.intcoords=[] # each coord is a tuple of 5 ints: coord type + 4 atom idxs
	natoms=o.natoms
	ageo=np.reshape(o.geo.copy(),(np.size(o.geo),1))
	# temporary lists
	valence=[] #store number of bonds for each atom
	centred=[] #store number angles this atoms is the centre of
	coordination=[]
	for i in range(natoms):
		coordination.append([])
		valence.append(0)
		centred.append(0)
	lbonds=[]  #indexes for bonds
	vbonds=[]  #lenghts for bonds
	langles=[] #indexes for angles
	vangles=[] #amplitude of angles
	loop=[]    #indexes of out of plane 
	voop=[]    #amplitudes for out of plane
	ltors=[]   #indexes for torsions
	vtors=[]   #amplitudes for torsions
	# add user-defined ICs
	if(useric!=[]):
		for i in range(len(useric)):
			if(useric[i][0]==1):
				lbonds.append(useric[i][1:])
				valence[useric[i][1]] += 1
				valence[useric[i][2]] += 1
				coordination[useric[i][1]].append(useric[i][2])
				coordination[useric[i][2]].append(useric[i][1])
			elif(useric[i][0]==2):
				langles.append(useric[i][1:])
			elif(useric[i][0]==3):
				loop.append(useric[i][1:])
			elif(useric[i][0]==4):
				ltors.append(useric[i][1:])
	# search for bonds
	for i in range(natoms):
		r1=Raddi[Symbols.index(o.symbol[i].lower())]
		for j in range(i+1,natoms):
			rl=(1.0+tol)*(r1+Raddi[Symbols.index(o.symbol[j].lower())])
			if ((bondLength(geo,[i,j])<rl)and([i,j] not in lbonds)):
				lbonds.append([i,j])
				valence[i] += 1
				valence[j] += 1
				coordination[i].append(j)
				coordination[j].append(i)
	#find triatomic angles between atomic bonds
	for i in range(len(lbonds)):
		set1=set(lbonds[i])
		for j in range(i+1,len(lbonds)):
			set2=set(lbonds[j])
			if (len(set1&set2)==1):
				tmp=list(set1^set2)
				tmp.sort()
				common=list(set1&set2)[0]
				candidate=[tmp[0],common,tmp[1]]
				centred[common] += 1
				if(Opts["doAutoSel"]):
					if((candidate not in langles)and(centred[common]<valence[common])):
						langles.append(candidate)
				else:
					if(candidate not in langles):
						langles.append(candidate)
	#find dihedrals (torsions) and out-of-plane (outs)
	#exclude if any of the angles is larger than 179 degs
	for i in range(len(langles)):
		if(angleAmp(geo,langles[i],True)>179.0):
			continue
		set1=set(langles[i])
		for j in range(i+1,len(langles)):
			if(angleAmp(geo,langles[j],True)>179.0):
				continue
			set2=set(langles[j])
			if (len(set1&set2)==2):
				tmp=list(set1^set2) # non-common atoms
				tmp2=list(set1&set2)# common atoms
				if(langles[i][1]==langles[j][1]): # out-of-plane 
					a2=langles[i][1] #reference
					tmplst=tmp+tmp2
					tmplst.pop(tmplst.index(a2)) #remove the common
					tmplst.sort()
					a1=tmplst[0]
					a3=tmplst[1]
					a4=tmplst[2]
					if(Opts['doOuts']):
						if(Opts['oopp']):
							teta=oopAmp(geo,[a1,a2,a3,a4])
							if((np.abs(teta-np.pi)<Opts['oopTol'])and([a1,a2,a3,a4] not in loop)):
								loop.append([a1,a2,a3,a4])
						else:
							if([a1,a2,a3,a4] not in loop):
								loop.append([a1,a2,a3,a4])
				# now the torsions...
				else:
					candidate=[]
					if((langles[i][1]==langles[j][0])and(langles[j][1]==langles[i][0])):
						candidate=[langles[i][2],langles[i][1],langles[j][1],langles[j][2]]
					elif((langles[i][1]==langles[j][0])and(langles[j][1]==langles[i][2])):
						candidate=[langles[i][0],langles[i][1],langles[j][1],langles[j][2]]
					elif((langles[i][1]==langles[j][2])and(langles[j][1]==langles[i][0])):
						candidate=[langles[i][2],langles[i][1],langles[j][1],langles[j][0]]
					elif((langles[i][1]==langles[j][2])and(langles[j][1]==langles[i][2])):
						candidate=[langles[i][0],langles[i][1],langles[j][1],langles[j][0]]
					if(Opts["doAutoSel"]):
						if((candidate!=[])and(candidate[1:]not in [h[1:] for h in ltors])):
							if(Opts['doTors']):
								ltors.append(candidate)
					else:
						if(candidate!=[]):
							if(Opts['doTors']):
								ltors.append(candidate)
	#calc S for bonds
	o.S=np.zeros((3*natoms,len(lbonds)+len(langles)+len(loop)+len(ltors)))
	o.intcoords=[]
	n=0
	for i in range(len(lbonds)):
		o.intcoords.append([1]+lbonds[i]+[0,0]) 
		sb=np.zeros(np.shape(geo))
		b0=bondLength(geo,lbonds[i])
		vbonds.append(b0)
		for j in range(np.size(geo,0)):
			for k in range(3):
				gp=np.copy(geo)
				gp[j,k]+=delta
				b1=bondLength(gp,lbonds[i])
				sb[j,k]=(b1-b0)/delta
		o.S[:,n]=np.reshape(sb,(1,np.size(sb)))
		n+=1
	# calc S for angles
	for i in range(len(langles)):
		o.intcoords.append([2]+langles[i]+[0])
		sb=np.zeros(np.shape(geo))
		phi0=angleAmp(geo,langles[i])
		vangles.append(np.rad2deg(phi0))
		if(phi0>178.0): #linear bend
			for j in [langles[i][1]]:
				for k in range(3):
					gp=np.copy(geo)
					gp[j,k]+= delta #(5.0*delta)
					phi1=angleAmp(gp,langles[i])
					sb[j,k]=(phi1-phi0)/(delta)
			for j in [langles[i][0],langles[i][2]]:
				for k in range(3):
					gp=np.copy(geo)
					gp[j,k]+=delta#*(np.sin(phi0))**2
					phi1=angleAmp(gp,langles[i])
					sb[j,k]=(phi1-phi0)/delta
		else:
			for j in [langles[i][1]]:
				for k in range(3):
					gp=np.copy(geo)
					gp[j,k]+= delta#*(np.sin(phi0))**2
					phi1=angleAmp(gp,langles[i])
					sb[j,k]=(phi1-phi0)/delta
			for j in [langles[i][0],langles[i][2]]:
				for k in range(3):
					gp=np.copy(geo)
					gp[j,k]+=delta#*(np.sin(phi0))**2
					phi1=angleAmp(gp,langles[i])
					sb[j,k]=(phi1-phi0)/delta
		#Scaling of the central displacement is recommended to
		#deal with linear bends
		o.S[:,n]=np.reshape(sb,(1,np.size(sb)))
		n+=1
	# calc S for outs
	for i in range(len(loop)):
		o.intcoords.append([3]+loop[i])
		sb=np.zeros(np.shape(geo))
		teta0=oopAmp(geo,loop[i])
		voop.append(np.rad2deg(teta0))
		#find how many atoms participate in only one bond
		aToMove=[]
		for j in loop[i]:
			nb=0
			for bond in lbonds:
				if(j in bond): nb += 1
			if(nb==1): aToMove.append(j)
		#if that fails, find which atom(s) has(ve) the least number of bonds
		if(len(aToMove)==0):
			rec=100
			for j in loop[i]:
				nb=0
				for bond in lbonds:
					if(j in bond): nb += 1
				if(nb==rec):
					aToMove.append(j)
				if(nb<rec):
					rec=nb
					aToMove=[j]
		for j in aToMove: # move only atoms participating in one bond
			for k in range(3):
				gp=np.copy(geo)
				gp[j,k]+= delta
				teta1=oopAmp(gp,loop[i])
				sb[j,k]=(teta1-teta0)/delta
		o.S[:,n]=np.reshape(sb,(1,np.size(sb)))
		n+=1
	# calc S for torsions
	for i in range(len(ltors)):
		sb=np.zeros(np.shape(geo))
		teta0=torsionAmp(geo,ltors[i])
		vtors.append(np.rad2deg(teta0))
		for j in [ltors[i][ii] for ii in [0,-1]]: # these two define the angle between planes
			for k in range(3):
				gp=np.copy(geo)
				gp[j,k]+= delta
				teta1=torsionAmp(gp,ltors[i])
				# account for torsion going across 0 or above 360
				if((teta0<0.2)and(teta1>5.0)):
					teta1 -= (2.0*np.pi)
				if((teta1<0.2)and(teta0>5.0)):
					teta1 -= (2.0*np.pi)
				sb[j,k]=(teta1-teta0)/delta
		o.intcoords.append([4]+ltors[i])
		o.S[:,n]=np.reshape(sb,(1,np.size(sb)))
		n+=1
	#clean-up S
	o.S=o.S[:,0:n]

## Analysis Functions ##

def VMP(of,s):
	""" Performs a simple projection of the normal modes 
	over the internal coordinates S stored in system s. Other inputs:
	- of: handle for the output file (for logging)"""
	of.write("\nStarting: Normal Mode Projection\n")
	if(len(s.intcoords)>len(s.vibrations)):
		of.write(" More internal coordinates than frequencies, expect\n")
		of.write(" some (possibly unwanted) redundancy in the results\n")
	for i in range(len(s.vibrations)):
		o=[]
		for j in range(len(s.intcoords)):
			o.append(np.dot(s.S[:,j],s.ADM[:,i])/np.linalg.norm(s.S[:,j]))
		s.vibrations[i].addVMP(o)
	of.write("\nEnding: Normal Mode Projection\n")

def VMLD(of,s):
	""" Performs a Linear Decomposition of the vibrational modes 
	over the internal coordinates stored in the system s.
	Inputs:
	- of: handle for the output file
	- s:  the system"""
	of.write("\nStarting: Vibrational Mode Linear Decomposition\n")
	if(len(s.intcoords)>len(s.vibrations)):
		of.write(" More internal coordinates than frequencies, expect\n")
		of.write(" some (possibly unwanted) redundancy in the results\n")
	for i in range(len(s.vibrations)):
		o=[]
		regressor=sklm.LinearRegression()
		regressor.fit(s.S,s.ADM[:,i])
		r2=np.corrcoef(s.ADM[:,i],regressor.predict(s.S))[0,1]**2
		exvar=skmt.explained_variance_score(s.ADM[:,i],regressor.predict(s.S))
		s.vibrations[i].addVMLD(regressor.coef_, r2, exvar)
	of.write("\nEnding: Vibrational Mode Linear Decomposition\n")

def VMBLD(of,s):
	""" Performs a Linear Decomposition of the vibrational modes 
	over the internal coordinates stored in the system s using Bayesian 
	Ridge Regression.
	Inputs:
	- of: handle for the output file
	- s:  the system"""
	of.write("\nStarting: Vibrational Mode Bayesian Linear Decomposition\n")
	if(len(s.intcoords)>len(s.vibrations)):
		of.write(" More internal coordinates than frequencies, expect\n")
		of.write(" some (possibly unwanted) redundancy in the results\n")
	for i in range(len(s.vibrations)):
		o=[]
		regressor=sklm.BayesianRidge(compute_score=True,n_iter=5000)
		regressor.fit(s.S,s.ADM[:,i])
		r2=np.corrcoef(s.ADM[:,i],regressor.predict(s.S))[0,1]**2
		exvar=skmt.explained_variance_score(s.ADM[:,i],regressor.predict(s.S))
		s.vibrations[i].addVMBLD(regressor.coef_, r2, exvar)
	of.write("\nEnding: Vibrational Mode Bayesian Linear Decomposition\n")

def VMARD(of,s):
	""" Performs a Linear Decomposition of the vibrational modes 
	using Bayesian regression with Automatic Relevance Determination
	over the internal coordinates stored in the system s.
	Inputs:
	- of: handle for the output file
	- s:  the system"""
	of.write("\nStarting: Vibrational Mode Automatic Relevance Determination\n")
	if(len(s.intcoords)>len(s.vibrations)):
		of.write(" More internal coordinates than frequencies, expect\n")
		of.write(" some (possibly unwanted) redundancy in the results\n")
	for i in range(len(s.vibrations)):
		o=[]
		regressor=sklm.ARDRegression(compute_score=True,n_iter=5000)
		regressor.fit(s.S,s.ADM[:,i])
		r2=np.corrcoef(s.ADM[:,i],regressor.predict(s.S))[0,1]**2
		exvar=skmt.explained_variance_score(s.ADM[:,i],regressor.predict(s.S))
		s.vibrations[i].addVMARD(regressor.coef_, r2, exvar)
	of.write("\nEnding: Vibrational Mode Automatic Relevance Determination\n")

## Main executable ##
if(__name__=='__main__'):
	# Set flags and filenames
	Linear=False # if true, internal coords will be 3N-5m instead of 3N-6
	Transition=False #if true, get the first mode and exclude the next 6 (or 5)
	if(len(sys.argv)<2):
		print("""Usage: %s [ Commands ] [ Options ] inputFile

Commands:
 --vmp     Perform Vibrational Mode Decomposition analysis.
 --novmp   Don't perform Vibrational Mode Decomposition (default).
 --vmld    Perform Vibrational Mode Linear Decomposition analysis.
 --novmld  Don't perform Vibrational Mode Linear Decomposition (default).
 --vmbld   Perform Vibrational Mode Bayesian Linear Decomposition analysis.
 --novmbld Don't perform Vibrational Mode Bayesian Linear Decomposition (default).
 --vmard   Perform Vibrational Mode ARD Decomposition analysis (default).
 --novmard Don't perform Vibrational Mode ARD Decomposition analysis.

Options:
 --linear         System is linear (3N-5 vibrational modes expected).
 --ts             System is a transition state 
                  (first vibrational mode included as reaction coordinate).
 --mwd            Use mass-weighted vibrational displacements (default).
 --nomwd          Don't use mass-weighted vibrational displacements.
 --mws            Use mass-weighted internal coordinate displacements.
 --nomws          Don't use mass-weighted internal coordinate displacements (default).
 --noouts         Don not generate out-of-plane internal coordinates
 --notors         Don not generate torsion internal coordinates
 --strictplanes   Restrict out-of-plane bending to co-planar atoms (default)
 --nostrictplanes Out-of-plane coordinates generated for all eligible 4 atom sets
 --ooptol f.ff    Tolerance (in degrees) for --strictplanes.
 --autosel        Automatic selection of internal coordinates.
 --addic FILE     Read additional or user-defined internal coordinates from
                  file FILE.
 --cut XX         Set the cutoff for presenting contributions:
                  'auto' - Automatic selection (default)
                  'all'  - All contributions are listed.
                  'd9'   - Always lists the top 90%% most important contributions
                  'q1'   - Always lists the top 25%% most important contributions
                  X      - List only contributions with relative weight above X%%
 --delta f.ff     Use this value for the discete computation of S.
 --tol XX         Tolerance (in %%) for determination of atomic connectivity.
 --input XX       Input format, and their expected sufixes:
                  'hess'      - Orca .hess file (default).
                  'g09'       - Gaussian09 output (log) file.
                  'mopac'     - MOPAC 2016 output (out) file.
                  'mopac2016' - MOPAC 2016 output (out) file.
 --vm XX          Animate vibrational mode XX 
 --ic XX          Animate internal coordinate XX 

 Format for additional coordinates (--addic):
 - Plain text file (extension is not relevant, but .ic is recommended)
 - Lines Containing a hash (#) are ignored as comments
 - One internal coordinate per line, as a tuple of characters and numbers:
   - First Entry: B A O, or T, for bond, angle, out-of-plane or torsion, respectively
	 - Following entries: the indexes of the atoms involved, starting with 1.
   - For B, you must define 2 atoms
   - For A, you must define 3 atoms, the second being the apex of the angle
   - For B, you must define 4 atoms, the second being the central atom
   - For B, you must define 4 atoms, the first and last being the extreme of the torsion
  Examples:
    B 1 2   -> Bond between atoms 1 and 2
    A 1 4 3 -> Angle formed betweem 4-1 and 4-3

\n"""%(sys.argv[0]))
		sys.exit(1)
	arg=sys.argv[1:-1]
	n=0
	inic=False
	invm=False
	ofn='va.out'
	useric=[]
	while(n<len(arg)):
		if(arg[n]=='--linear'):
			inic=False
			invm=False
			Opts['isLinear']=True
		elif(arg[n]=='--ts'):
			inic=False
			invm=False
			Opts['isTState']=True
		elif(arg[n]=='--mwd'):
			inic=False
			invm=False
			Opts['doMWD']=True
		elif(arg[n]=='--nomwd'):
			inic=False
			invm=False
			Opts['doMWD']=False
		elif(arg[n]=='--mws'):
			inic=False
			invm=False
			Opts['doMWS']=True
		elif(arg[n]=='--nomws'):
			inic=False
			invm=False
			Opts['doMWS']=False
		elif(arg[n]=='--autosel'):
			inic=False
			invm=False
			Opts['doAutoSel']=True
		elif(arg[n]=='--noouts'):
			inic=False
			invm=False
			Opts['doOuts']=False
		elif(arg[n]=='--strictplanes'):
			inic=False
			invm=False
			Opts['oopp']=True
		elif(arg[n]=='--nostrictplanes'):
			inic=False
			invm=False
			Opts['oopp']=False
		elif(arg[n]=='--notors'):
			inic=False
			invm=False
			Opts['doTors']=False
		elif(arg[n]=='--addic'):
			useric=readUserIC(arg[n+1])
			n += 1
		elif(arg[n]=='--cut'):
			inic=False
			invm=False
			if(arg[n+1]=='auto'):
				Opts['cut']='auto'
			elif(arg[n+1]=='d9'):
				Opts['cut']='d9'
			elif(arg[n+1]=='q1'):
				Opts['cut']='q1'
			elif(arg[n+1]=='all'):
				Opts['cut']='all'
			else:
				Opts['cut']='val'
				Opts['cutval']=float(arg[n+1])/100.0
			n += 1
		elif(arg[n]=='--delta'):
			inic=False
			invm=False
			Opts['delta']=float(arg[n+1])
			n += 1
		elif(arg[n]=='--ooptol'):
			inic=False
			invm=False
			Opts['oopTol']=np.deg2rad(float(arg[n+1]))
			n += 1
		elif(arg[n]=='--tol'):
			inic=False
			invm=False
			Opts['tol']=float(arg[n+1])/100.0
			n += 1
		elif(arg[n]=='--input'):
			inic=False
			invm=False
			if(arg[n+1]=='hess'):
				Opts['input']='OrcaHess'
			elif(arg[n+1]=='g09'):
				Opts['input']='G09OUT'
			elif(arg[n+1]=='mopac' or arg[n+1]=='mopac2016'):
				Opts['input']='MOPAC2016'
			else:
				print('ERROR: Input format not implemented!\n')
				sys.exit(1)
			n += 1
		elif(arg[n]=='--vm'):
			inic=False
			invm=True
		elif(arg[n]=='--ic'):
			inic=True
			invm=False
		## commands ##
		elif(arg[n]=='--vmp'):
			inic=False
			invm=False
			Opts['doVMP']=True
		elif(arg[n]=='--novmp'):
			inic=False
			invm=False
			Opts['doVMP']=False
		elif(arg[n]=='--vmld'):
			inic=False
			invm=False
			Opts['doVMLD']=True
		elif(arg[n]=='--novmld'):
			inic=False
			invm=False
			Opts['doVMLD']=False
		elif(arg[n]=='--vmbld'):
			inic=False
			invm=False
			Opts['doVMBLD']=True
		elif(arg[n]=='--novmbld'):
			inic=False
			invm=False
			Opts['doVMBLD']=False
		elif(arg[n]=='--vmard'):
			inic=False
			invm=False
			Opts['doVMARD']=True
		elif(arg[n]=='--novmard'):
			inic=False
			invm=False
			Opts['doVMARD']=False
		elif(inic):
			Opts['anic'].append(int(arg[n]))
		elif(invm):
			Opts['aniMode'].append(int(arg[n]))
		n += 1
	ifn=sys.argv[-1]
	#Create name/basename for output file
	if(Opts['input']=='OrcaHess'):
		ofn=ifn[:-5]
	if(Opts['input']=='G09OUT'):
		ofn=ifn[:-4]
	if(Opts['input']=='MOPAC2016'):
		ofn=ifn[:-4]
	of=open(ofn+'.nma','w')
	of.write("""###############################################################
#                                                             #
#   vibAnalysis - version 1.20 beta                           #
#   A set of tools to analyse vibrational modes in terms of   #
#   localized internal coordinates.                           #
#                                                             #
#   (c) Filipe Teixeira, 2017                                 #
#   filipe _dot_ teixeira _at_ fc _dot_ up _dot_ pt           #
#                                                             #
###############################################################\n""")
	# Read Geometry, Masses, Frequencies and Vibrational Mode Displacements
	if(Opts['input']=='OrcaHess'):
		of.write("\nOpening Hess file: %s\n\n"%(ifn))
		system=readHess(ifn)
	elif(Opts['input']=='G09OUT'):
		of.write("\nOpening Gaussian09 output file: %s\n\n"%(ifn))
		system=readG09log(ifn)
	elif(Opts['input']=='MOPAC2016'):
		of.write("\nOpening MOPAC 2016 output file: %s\n\n"%(ifn))
		system=readMopac2016(ifn)
	## If MW, mass-weight the normal modes
	# Orca's hess is not mass-weighted, but normalized
	# Gaussian's log is not mass-weighted, but normalized
	if(Opts['doMWD']):
		of.write("\nMass-weighting the atomic displacements...\n\n") #OLD
		system.massWeightVibrations()
		system.normalizeVibrations()
	system.sortVibrations()
	system.makeADM()
	system.checkADM()
	# Generate quasi-redundant internal coordinated
	of.write("\nGenerating Internal Coordinates...\n")
	makeIC(system,useric)
	# normalize S
	if(Opts['doMWS']):
		system.massWeightS()
	system.normalizeS()
	of.write("Internal Coordinates Generated: %d\n"%(len(system.intcoords)))
	# print list of internal coordinates
	of.write("""\nThis is the list of Internal Coordinates generated for
this run. Please use these indexes if you whish to animate
any one of these coordinates. If you added some user-defined
coordinates, please be sure to re-run this program with the
same added coordinates.\n""")
	punchIC(of,system)
	## Animate modes?
	if(len(Opts['aniMode'])>0):
		of.write("\n")
		for mode in Opts['aniMode']:
			tfn="%s.v%03d.xyz"%(ofn,mode)
			of.write("Animating mode %d to file: %s\n"%(mode,tfn))
			animateMode(tfn,system,mode-1)
		of.write("\n")
	if(len(Opts['anic'])>0):
		of.write("\n")
		for icidx in Opts['anic']:
			tfn="%s.i%03d.xyz"%(ofn,icidx)
			of.write("Animating internal coordinate %d to file: %s\n"%(icidx,tfn))
			animateIC(tfn,system,icidx-1)
		of.write("\n")
	## Do the analysis
	if(Opts['doVMP']):
		VMP(of,system)
	if(Opts['doVMLD']):
		VMLD(of,system)
	if(Opts['doVMBLD']):
		VMBLD(of,system)
	if(Opts['doVMARD']):
		VMARD(of,system)
	## Print the final analysis
	printResults(of,system)
	## Clean up
	of.close()

