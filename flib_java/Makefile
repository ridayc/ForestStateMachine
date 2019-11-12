flib = flib
source = $(flib)/
targetc = ../flib_class/
target =$(targetc)$(flib)/

command = javac --release 8 -d "$(targetc)"

# Folders
MATH = math/
MATHS=$(source)$(MATH)
MATHT=$(target)$(MATH)
RF = algorithms/randomforest/
RFS=$(source)$(RF)
RFT=$(target)$(RF)

# File Names
VCJ = $(MATHS)VectorConv.java
VCC = $(MATHT)VectorConv.class
VFJ = $(MATHS)VectorFun.java
VFC = $(MATHT)VectorFun.class
DNJ = $(RFS)DecisionNode.java
DNC = $(RFT)DecisionNode.class
TNJ = $(RFS)TreeNode.java
TNC = $(RFT)TreeNode.class

all : \
$(VCC) \
$(VFC) \
$(DNC) \
$(TNC) \

.PHONY: all
$(VCC) : $(VCJ)
	$(command) $(VCJ)
	
$(VFC) : $(VFJ)
	$(command) $(VFJ)
	
$(DNC) : $(DNJ)
	$(command) $(DNJ)
	
$(TNC) : $(TNJ)
	$(command) $(TNJ)