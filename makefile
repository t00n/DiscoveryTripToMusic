TOZIP=src/makefile $(wildcard src\/*.py) src/crossvalidate.pl makefile readme.md requirements.txt data/

ZIPFILE=DiscoveryTripToMusic.zip

$(ZIPFILE):
	zip -r $@ $(TOZIP)

mrproper:
	rm $(ZIPFILE)

test:
	make -C src test

validation:
	make -C src validation

features:
	make -C src features

clean:
	make -C src clean