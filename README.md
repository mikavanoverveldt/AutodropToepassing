# Autodrop Toepassing

Computervisie toepassing voor automatische kwaliteitsinspectie van snoepverwerkingsproducten. Detecteert validiteit van producten op basis van kleur en grootte.

## Projectstructuur

```
Software/
├── CandyDetector/          # Productie applicatie (Linux)
├── Camera_Gray/            # Sample: OpenCV basics
├── DahengLinux/            # Sample: Galaxy SDK introductie
├── WindowsTest/            # Sample: C++ variant
└── autodrop v5/            # Sample: Calibratietoepassing (Windows)
```

# Setup:
### Download OpenCV en dependencies
Voor Fedora:
```
sudo dnf install opencv opencv-devel gcc-c++ cmake pkg-config
```

Voor andere distro's, gebruik de eigen package manager.

### Download Galaxy Camera SDK
Download url: https://hs.va-imaging.com/hubfs/Supportshare/Daheng%20SDK/Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231.zip

Unzip deze map:
```
/FOLDER/OF/CHOICE/Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231/
```

Ga in deze map. Voer Galaxy_camera.run uit als `sudo`  (!) 

```
sudo ./Galaxy_camera.run
```
Volg het installatieprogramma

Doe hierna een reboot (`sudo reboot`)


#### Galaxy SDK library pad toevoegen (ldconfig)

Voeg de Galaxy SDK library directory toe aan de dynamic linker search path en herlaad de cache. Pas het pad hieronder aan naar jouw installatiepad indien nodig:

```bash
echo "FOLDER/OF/CHOICE/Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231/Galaxy_camera/lib/x86_64" | sudo tee /etc/ld.so.conf.d/galaxy-sdk.conf
sudo ldconfig
```

Je wordt mogelijk gevraagd om je `sudo`-wachtwoord. Controleer daarna of de Galaxy-libraries zichtbaar zijn:

```bash
ldconfig -p | grep -i galaxy
```

Als een van de commando's een resultaat toont, is het library-pad succesvol toegevoegd.


## CandyDetector Starten

**Vereisten:**
- OpenCV 4.x
- Galaxy SDK (zie hierboven)
- Daheng MER-133-54U3C camera aangesloten

**Build & Run:**
```bash
cd Software/CandyDetector
make              # Compileer
make run          # Compileer en voer uit
make clean        # Opschonen
```

**Werking:**
- 8-staps verwerkingspijplijn: grijs → HSV → kleurbereiken → morfologie → contourdetectie
- Toont 10 visualisatievensters met real-time parameters
- Trackbars voor live parameter aanpassing
- Groen = geldig, rood = ongeldig
- Ongeveer 30 FPS

**Validatiecriteria:**
- Geldig: rode pixels < 20% EN oppervlakte > 1500px
- Ongeldig: rode pixels ≥ 20% OF oppervlakte < 1500px

## Dependencies

**Linux:**
- GCC/G++ met C++17
- OpenCV 4.x
- Galaxy SDK 2.4.2507.9231

