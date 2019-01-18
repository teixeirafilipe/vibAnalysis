# vibAnalysis
Tools for performing vibrational analysis on molecular systems.

## About vibAnalysis
**vibAnalysis** is a tool for aiding the interpretation of vibrational spectra
using the Vibrational Mode Decomposition (VMD) paradigm. This is an alternative
approach to the well-established Potential Energy Decomposition (PED) scheme
implemented in software packages such as [VEDA](http://smmg.pl/software/veda). 

Under VMD, the movement associated with a vibrational mode is decomposed into a
set of redundant internal coordinates. For each vibrational mode, the most
prominent internal coordinates describing the atomic motions are listed,
together with their respective weights. In **vibAnalysis**, we implemented three
different "flavours" of VMD, which differ in the way such weights are
determined:
1. Vibrational Mode Projection (VMP)
1. Vibrational Mode Linear Decomposition (VMLD), and
1. Vibrational Mode Automatic Relevance Determination (VMARD)

The latter of which is the default method used by **vibAnalysis**. VMP and VMLD
are simple methods based on vector projection and linear least-squares
regression, respectively. In its turn, VMARD uses the [Automatic Relevance
Determination](http://scikit-learn.org/stable/modules/linear_model.html#automatic-relevance-determination-ard)
(ARD) method, which is a variant of Bayesian Ridge Regression to better estimate
the most prominent internal coordinates. This results on a much clearer picture
of the internal coordinates describing the atomic motions.

Currently, **vibAnalysis** can perform VMP, VMLD, VMARD, as well as animate
vibrational motions and displacements along internal coordinates.

## System Requirements
**vibAnalysis** is written in Python 3, and should work in any modern system. Its
only requirements are:
* [Numpy](http://www.numpy.org/)
* [Scikit-learn](http://scikit-learn.org/stable/) 

**vibAnalysis** has the ability to export the animation of vibrational modes and
displacements along internal coordinates using the xyz format which can be
easily opened using [Molden](http://www.cmbi.ru.nl/molden/).

## How to Use
Assuming you have downloaded the _va.py_, have permissions to execute it, and it
is in your executable PATH. If you run _va.py_ without any argument, it will
print its help on the terminal.

```
Usage: ./va.py [ Commands ] [ Options ] inputFile

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
                  'd9'   - Always lists the top 90% most important contributions
                  'q1'   - Always lists the top 25% most important contributions
                  X      - List only contributions with relative weight above X%
 --delta f.ff     Use this value for the discete computation of S.
 --tol XX         Tolerance (in %) for determination of atomic connectivity.
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
   - For O, you must define 4 atoms, the second being the central atom
   - For T, you must define 4 atoms, the first and last being the extreme of the torsion
  Examples:
    B 1 2   -> Bond between atoms 1 and 2
    A 1 4 3 -> Angle formed betweem 4-1 and 4-3

```

### Basic Usage
**vibAnalysis** can read from either the output of Gaussian09 or from the hess
file generated by Orca (the latter being the default). The defaults are sensible
enough for most uses. Thus, it can be invoked only as:

```
va.py myFreqRun.hess
```

for analysing the hess file from Orca. The results will be punched onto a file
named _myFreqRun_.nma. If you prefer to use Gaussian for your calculations, you
can use

```
va.py --input g09 myFreqRun.log
```

as before, the results will be saved on a file named  _myFreqRun_.nma. 

### Animating Vibrational Modes and Internal Coordinates

You can animate one or more vibrational modes using **vibAnalysis**, just use
the --vm option followed by a list of the modes to animate, for example:

```
./va.py --vm 11 12 acoh-hess.hess
```

will make the VMARD analysis and also punch the animation of vibrational modes
11 and 12 onto files named acoh-hess.v011.xyz and acoh-hess.v012.xyz. The
numbering of the vibrational modes corresponds to those on the nma file, which
may or may not correspond to the ordering read in the files generated by the
Quantum chemistry packages.

In a similar manner, the motion along a single internal coordinate can be
animated using the --ic option, for example:

```
./va.py --ic 11 12 acoh-hess.hess
```

will punch the animation along internal coordinates 11 and 12 onto files
acoh-hess.i011.xyz and acoh-hess.i012.xyz, respectively. These files can be used
to inspect the displacement along each internal coordinate, or can be used to
perform Potential Energy Surface (PES) scans for the parametrisation of
molecular force fields.

## How to Cite
The VMARD method is described in _J. Chem. Theory Comput_, **2019**, _15_(1),
456-470. If you wish to publish your work using **vibAnalysis**, please please
cite the above paper, using the following BibTeX entry:

```bibtex
@Article{Teixeira2018VMARD,
  author    = {Filipe Teixeira and M. Nat{\'{a}}lia D. S. Cordeiro},
  title     = {Improving Vibrational Mode Interpretation Using Bayesian Regression},
  journal   = {J. Chem. Theory Comput.},
  year      = {2019},
  volume    = {15},
  number    = {1},
  pages     = {456--470},
  month     = {dec},
  doi       = {10.1021/acs.jctc.8b00439},
  publisher = {American Chemical Society ({ACS})}
}

```

In addition to this, you may also want to put a reference to this gitHub
repository, as the code evolves and new implementation features are added. The
following BibTeX entry should provide you with all the needed information:

```bibtex
@Electronic{TeixeiraVibAnal,
  author    = {Filipe Teixeira},
  title     = {VibAnalysis - Tools for performing vibrational analysis on molecular systems},
  year      = {2017},
  date      = {2017-07-04},
  url       = {https://github.com/teixeirafilipe/vibAnalysis},
  urldate   = {2017-07-04}
}
```

