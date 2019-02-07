NVCC = nvcc

# Target -> Dependency -> Recipe(defines compilation directive)
# -I -> input directory for project..."./" in this case
#-o -> to generate executable
helloCUDA: headers.h kernel.cu host.cu  
	$(NVCC) -I ./ kernel.cu host.cu -o ass_1

# command "make run" can run code in single gpu system
run:
	./ass_1

clean:
	rm -rf ass_1
