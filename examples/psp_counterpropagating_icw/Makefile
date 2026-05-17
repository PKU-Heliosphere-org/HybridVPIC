# Define shell
SHELL = csh

# Directory containing the HybridVPIC deck compiler. Override on other
# machines with:
#   make PROJECTDIR=/path/to/hybridVPIC/bin
PROJECTDIR ?= ../../build/bin

all:
	${PROJECTDIR}/vpic psp_counterpropagating_icw.cxx

translate:
	mpif90 -o translate_psp_icw translate_psp_icw.f90

cleanall: 	
#	cd translate; make clean 
	-rm data/*
	-rm -rf hydro
	-rm -rf particle
	-rm -rf restart*
	-rm *.Linux
	-rm *.bin
	-rm *.vpc
	-rm *.btr
	-rm -rf rundata
	-rm -rf window
	-rm -rf fields
	-rm -rf spectrum
	-rm energy_spectrum
	-rm *.cy
	-rm *.op
	-rm *.ppu
	-rm outfile
	-rm info
	-rm *.eps
	-rm *.mpg
	-rm core.*
	-rm *~




