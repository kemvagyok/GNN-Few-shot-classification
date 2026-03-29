import torch

def dataAboutSpaceGPU():
    if torch.cuda.is_available():
        # Lekérdezzük a szabad és a teljes memóriát bájtokban az aktuális (0-s) GPU-n
        free_mem, total_mem = torch.cuda.mem_get_info()
        
        # Bájtok átváltása Gigabájtra (GB)
        free_mem_gb = free_mem / (1024 ** 3)
        total_mem_gb = total_mem / (1024 ** 3)
        occupied_mem_gb = total_mem_gb - free_mem_gb
        
        print(f"Összes GPU memória: {total_mem_gb:.2f} GB")
        print(f"Szabad GPU memória: {free_mem_gb:.2f} GB")
        print(f"Foglalt GPU memória: {occupied_mem_gb:.2f} GB")
    else:
        print("CUDA nem elérhető.")