# PyTorch & Lustre

  Ispod se nalaze datoteke potrebne za testiranje brzine DataLoader funkcije u
  sklopu sučelja Accelerate za PyTorch. Skripte se sastoje od kreiranja matrice
  s `(100*10000) * 2048` elemenata, koje se učitavaju u raznim formatima s Lustre
  datotečnog sustava. Skripte se sastoje od dva dijela koja se moraju izvesti
  redom:

  1. `generate_test_dataset*` - skripte za generiranje podataka
  2. `run_test_dataset*` - skripte za mjerenje brzine učitavanja

  Gdje `*.sh` odgovaraju `bash` skriptama koje pokreću istoimenu skriptu
  `python` i mogu se direktno pozvati ili podnijeti kao posao u PBS-u.

## Inačice

  1. `run_test_dataset.py` - originalna verzija u kojoj postoji `100` direktorija
      s po `10000` datoteka `*.npy` s jednim retkom
  1. `run_test_dataset_squashfs.py` - verzija u kojoj se prethodno generirane
      datotke komprimiraju u `squashfs` image i bindaju unutar kontejnera
  1. `run_test_dataset_squashfs_singledir.py` - slično kao i prošla, no ovaj put
     sve datoteke u jednom direktoriju
  1. `run_test_dataset_memory.py` - verzija u kojoj se podaci učitavaju u
     memoriju
  1. `run_test_dataset_distributed.py` - distribuirana verzija (za jedan čvor)
     gdje svaki proces učitava datoteke u memoriju, u ovisnosti o broju
     grafičkih jezgri
