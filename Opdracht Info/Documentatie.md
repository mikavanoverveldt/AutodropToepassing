## Camera info
Camera:  MER-133-54U3C\
Resulutie: 1280 x 960\
Max FPS: 30

## Lens info
brandpuntafstand: 16 mm\
f-stop: 1:1.4/16 mm

## Eisen:
Inspectiegebied: 100 mm x 100 mm\
Inspectie maximaal 2 seconde\

## Berekende hoogte:
f = 16mm
FOV = 100 x 100 mm

Sensorbreedte = 1280 × 3.75 µm = 4.800 mm
Sensorhoogte  =  960 × 3.75 µm = 3.600 mm

Volledige Veld moet in de hoogte vallen, 3.6mm

$$wd = \frac{100 \text{ mm} \cdot 16 \text{ mm}}{3,6 \text{ mm}} + 16 \text{ mm}$$

$$wd = \frac{1600}{3,6} + 16$$

$$wd = 444,44 \text{ mm} + 16 \text{ mm}$$

**$$wd = 460,44 \text{ mm}$$**

Afstand is 46 cm



## Voorgestelde pipeline:
Input image --> Gaussian blur --> Naar HSV colorspace --> Blob mask voor snoepjes --> Morph close --> Morph Open --> findContours 

Voor iedere gevonden contour:
Enclosing Circle --> bereken area voor gebroken anders -->

Bereken gemiddelde kleur --> check kleur valid/fail --> Teken output over input --> Laat input zien