TOZIP=src/makefile $(wildcard src\/*.py) src/crossvalidate.pl makefile readme.md requirements.txt data/

project.zip:
	zip -r $@ $(TOZIP)

mrproper:
	rm project.zip

test:
	make -C src test

clean:
	make -C src clean