# Félig felügyelt tanulás/Semi-supervised learning
2025/26 Diplomaterv
FELADATKIÍRÁS
A felügyelt tanulási módszerek hatékony működéséhez nagyszámú címkézett adatra
van szükség, melyek előállítása költséges és erőforrás-igényes folyamat. A diplomamunka
célja egy olyan megközelítés kidolgozása, amely képes gráf neurális hálózattal osztályozási
feladatokat megoldani, osztályonként csupán néhány címkézett minta felhasználásával. A
tervezett módszer első lépése egy enkóder bevezetése az adatok vektros reprezentációjá-
nak elkészítéséhez. Ezt követően a látens térben felépíthető egy hasonlósági gráf, melyben
mindegyik csúcs egy-egy adatpontot reprezentál. Végül egy gráf neurális hálózat feladata
a címkézett és a címkézetlen adatok együttes felhasználása az optimális osztályozás eléré-
séhez. A vizsgálatok különböző adathalmazokon (pl. tabuláris, képi, szöveges) zajlanak,
azzal a céllal, hogy a tervezett módszer pontossága minél jobban megközelítse az adatbázis
teljes címkézése mellett elérhető szintet
A hallgató által kidolgozandó részfeladatok:
• Készítsen irodalmi áttekintést a few-shot tanulási módszerekről és a gráf neurális
hálózatokról.
• Készítse elő a kiválasztott adathalmazokat és a megfelelő enkóder modelleket a kü-
lönböző adattípusok (pl. képek, szövegek) vektorizálására.
• Dolgozzon ki módszereket a hasonlósági gráf felépítésére az enkóderek látens terében.
• Tanítson gráf neurális hálózatokat a few-shot osztályozási feladatokra.
• Értékelje ki az eredményeket a különböző adathalmazokon, összehasonlítva a kidol-
gozott módszereket a teljesen címkézett adatokon tanított modellekkel.
• Dokumentálja az eredményeket és publikálja az implementációt GitHub-on, mely
lehetővé teszi a teljes reprodukálhatóságot.
