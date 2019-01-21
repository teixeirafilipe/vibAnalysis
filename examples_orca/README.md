# Orca Examples

## Water molecule
Input, output, hess and xyz files conccerning the water optimisation and
frequency analysis at the B3LYP/Def2-SVP level of theory, using orca are
supplied with the `h2o-opthess` prefix. Subsequent vibrational analysis using
VMARD is executed using the following command:

```
$ va.py h2o-opthess.hess
```

## Methanol Molecule
Similar to the water example, VMP, VMLD and VMARD analysis are also carried out
using vibAnalysis. All files pertaining to this example share he `meoh-opthess`
prefix, and vibrational analysis was carried out using the following invocation
of vibAnalysis:

```
$ va.py --vmp --vmld meoh-opthess.hess
```

As can be seen in the example output from vibAnalysis
(meoh-opthess.mna_example), both VMP and VMLD fail to provide a good description
of the vibrations. This is particularly evident from the composition of the
vibrational mode with the highest wave number (3822 cm-1):

```
Coordintate          Weights (%)
                 VMP    VMLD  VMARD
BOND O2 H6      51.2    22.2   94.9
ANGLE O2 C1 H3  12.9    12.2    -- 
ANGLE C1 O2 H6  11.3     --     --
BOND  C1 O2      9.1     --     --
ANGLE H4 C1 H5   --     13.4    --
ANGLE H3 C1 H5   --     12.7    --
ANGLE O2 C1 H5   --     12.7    --
ANGLE O2 C1 H4   --     12.6    --
ANGLE H3 C1 H4   --     12.5    --
```

That being said, VMP still performs reasonably better than VMLD, giving the
overall description that this mode is largely a stretching motion along the
O2-H6 bond.

## Ethane
TBA

## Difluoroethene (_cis-_, _trans-_ and _gem-_ isomers)
TBA

## Acetic Acid
TBA

## Deuterated Acetic Acid
TBA

## Acetic Acid Dimer
TBA

## Benzene
TBA

## Cyclobutanone
TBA

## Aza-Diels Alder Transition State
Using the prefix `ada`, four reaction chanels.


