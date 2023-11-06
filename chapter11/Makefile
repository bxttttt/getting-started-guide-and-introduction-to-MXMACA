DEBUG ?= 0
MCCL ?=0
MCCLCMMD = -D_USE_MCCL -lmccl

ifeq ($(DEBUG), 0)
ifeq ($(MCCL),0)
simple2DFD_rls: simple2DFD.cpp
	mxcc -x maca -O3 ./simple2DFD.cpp -I./ -o ./build/$@
else 
simple2DFD_rls_mccl: simple2DFD.cpp
	mxcc -x maca -O3 ./simple2DFD.cpp  $(MCCLCMMD) -I./ -o ./build/$@
	@echo Useing mccl now!
endif
else
ifeq ($(MCCL),0)
simple2DFD_dbg: simple2DFD.cpp
	mxcc -x maca -g -G ./simple2DFD.cpp  -I./ -o ./build/$@
else
simple2DFD_dbg_mccl: simple2DFD.cpp
	mxcc -x maca -g -G ./simple2DFD.cpp  $(MCCLCMMD)  -I./ -o ./build/$@
	@echo Useing mccl now!
endif
endif

clean:
	rm -f ./build/simple2DFD_*

