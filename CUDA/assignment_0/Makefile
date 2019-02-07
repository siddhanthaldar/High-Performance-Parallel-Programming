NVCC = nvcc

# Target -> Dependency -> Recipe(defines compilation directive)
# -I -> input directory for project..."./" in this case
#-o -> to generate executable
helloCUDA: headers.h kernel.cu host.cu  
	$(NVCC) -I ./ kernel.cu host.cu -o helloCUDA

# command "make run" can run code in single gpu system
run:
	./helloCUDA

clean:
	rm -rf helloCUDA
