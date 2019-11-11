flib = flib
source = $(flib)/
targetc = ../flib_class/
target =$(targetc)$(flib)/

command = javac --release 8 -d "$(targetc)"

MATH = math/
MATHS=$(source)$(MATH)
MATHT=$(target)$(MATH)

all : \
$(MATHT)VectorConv.class \
$(MATHT)VectorFun.class

.PHONY: all
$(MATHT)VectorConv.class : $(MATHS)VectorConv.java
	$(command) $(MATHS)VectorConv.java
	
$(MATHT)VectorFun.class : $(MATHS)VectorFun.java
	$(command) $(MATHS)VectorFun.java