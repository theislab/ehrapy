ifeq ($(OS),Windows_NT)
include makefiles/Windows.mk
else
include makefiles/Linux.mk
endif
