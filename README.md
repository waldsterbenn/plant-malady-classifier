# Træning af et convolutional network.

Systemet bruger et eksisterende datasæt med billeder af blade fra raske og syge planter. Billederne i datasættet er allerede katagoriseret, så det her program træner et neuralt netværk for at lære det at genkende syge og raske blade. Og identificere hvad, om noget, der er i vejen med planten. Så det er supervised deep learning.

[See colab notebook](Plant-malady-identifier.ipynb)

## Programmet gør følgende:

1. Hent datasæt fra Kaggle. Det er et simpelt klassificerings dataset hvor folder navne er klasser, som indeholder billeder af blade i hver kategroi.
1. Del datasættet op 80/20, så man har noget data at teste med. Indfører også lidt støj i billederne for at undgå overfitting under træning.
1. Lav CNN model med 4 convoluting layers og dense layers med dropout og softmax output layer.
1. Træn netværket i x antal epochs, med et par mekanismer til at stoppe træning tidligt og undgå lokale plateauer.
1. Til sidst er der en testkørsel og visning af træning og validerings tab.

Kører bedst på A100 GPU $$$.

Aside: _Billederne kunne sådan set være alt muligt andet, men nu kan jeg godt lide planter. Man kunne snitte modellen til så den kan køre på en telefon, så kunne man lave en app der kunne gøre netværket mere anvendeligt i praksis. Eller måske bare køre det på en webserver._

Datasæt:
[PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease/data)
