flib = flib
source = $(flib)/
targetc = ../flib_class/
target =$(targetc)$(flib)/

command = javac --release 8 -d "$(targetc)" -cp "$(targetc)"

# Folders
ALGO = algorithms/
ALGOS=$(source)$(ALGO)
ALGOT=$(target)$(ALGO)
MATH = math/
MATHS=$(source)$(MATH)
MATHT=$(target)$(MATH)
RAND = $(MATH)random/
RANDS=$(source)$(RAND)
RANDT=$(target)$(RAND)
RF = $(ALGO)randomforest/
RFS=$(source)$(RF)
RFT=$(target)$(RF)

# File Names
RSJ = $(MATHS)RankSort.java
RSC = $(MATHT)RankSort.class
SPJ = $(MATHS)SortPair.java
SPC = $(MATHT)SortPair.class
SP2J = $(MATHS)SortPair2.java
SP2C = $(MATHT)SortPair2.class
VCJ = $(MATHS)VectorConv.java
VCC = $(MATHT)VectorConv.class
VFJ = $(MATHS)VectorFun.java
VFC = $(MATHT)VectorFun.class
SHJ = $(RANDS)Shuffle.java
SHC = $(RANDT)Shuffle.class
DNJ = $(RFS)DecisionNode.java
DNC = $(RFT)DecisionNode.class
TNJ = $(RFS)TreeNode.java
TNC = $(RFT)TreeNode.class

all : \
$(RSC) \
$(SPC) \
$(SP2C) \
$(VCC) \
$(VFC) \
$(SHC) \
$(DNC) \
$(TNC) \

.PHONY: all

$(RSC) : $(RSJ) $(SPC) $(VCC)
	$(command) $(RSJ)

$(SPC) : $(SPJ)
	$(command) $(SPJ)

$(SP2C) : $(SP2J)
	$(command) $(SP2J)

$(VCC) : $(VCJ)
	$(command) $(VCJ)
	
$(VFC) : $(VFJ)
	$(command) $(VFJ)
	
$(SHC) : $(SHJ) $(VFC)
	$(command) $(SHJ)
	
$(DNC) : $(DNJ)
	$(command) $(DNJ)
	
$(TNC) : $(TNJ)
	$(command) $(TNJ)