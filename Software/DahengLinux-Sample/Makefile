SDK_ROOT  := /home/mikavanoverveldt/Avans/Vision/Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231/Galaxy_camera
SDK_INC   := $(SDK_ROOT)/inc
SDK_LIB   := $(SDK_ROOT)/lib/x86_64

CXX       := g++
CXXFLAGS  := -std=c++17 -Wall -O2 \
             -I$(SDK_INC) \
             $(shell pkg-config --cflags opencv4)

LDFLAGS   := -L$(SDK_LIB) -Wl,-rpath,$(SDK_LIB) \
             -lgxiapi \
             $(shell pkg-config --libs opencv4) \
             -lpthread

TARGET    := DahengLinux
SRC       := DahengLinux.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run
