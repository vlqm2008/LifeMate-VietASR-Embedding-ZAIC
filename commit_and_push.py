import subprocess
import ctypes
from ctypes import wintypes
import sys
import os

git_exe = r"C:\Users\Admin\AppData\Local\GitHubDesktop\app-3.6.9\resources\app\git\cmd\git.exe"
repo_dir = r"D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC"

# 1. Git add
print("Running git add -A...")
subprocess.run([git_exe, "-C", repo_dir, "add", "-A"], check=True)

# 2. Git commit
print("Running git commit...")
msg = "Finalize Camera-Ready FJCAI 2027: Strict Sequential Citations [1..24], Authentic References, Exact 6-Page IEEE Layout, Zero-AI Graphics"
res = subprocess.run([git_exe, "-C", repo_dir, "commit", "-m", msg], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("Stderr:", res.stderr)

# 3. Read token from Windows Credential Manager
class CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ('Flags', wintypes.DWORD),
        ('Type', wintypes.DWORD),
        ('TargetName', wintypes.LPWSTR),
        ('Comment', wintypes.LPWSTR),
        ('LastWritten', wintypes.FILETIME),
        ('CredentialBlobSize', wintypes.DWORD),
        ('CredentialBlob', ctypes.POINTER(ctypes.c_char)),
        ('Persist', wintypes.DWORD),
        ('AttributeCount', wintypes.DWORD),
        ('Attributes', ctypes.c_void_p),
        ('TargetAlias', wintypes.LPWSTR),
        ('UserName', wintypes.LPWSTR),
    ]

advapi32 = ctypes.WinDLL('advapi32.dll')
CredReadW = advapi32.CredReadW
CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(CREDENTIAL))]
CredReadW.restype = wintypes.BOOL
CredFree = advapi32.CredFree
CredFree.argtypes = [ctypes.c_void_p]

pCred = ctypes.POINTER(CREDENTIAL)()
token = None
username = None

for target in ["git:https://github.com", "LegacyGeneric:target=git:https://github.com"]:
    if CredReadW(target, 1, 0, ctypes.byref(pCred)):
        cred = pCred.contents
        raw = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
        token = raw.decode('utf-16le', errors='ignore').strip('\x00')
        username = cred.UserName
        CredFree(pCred)
        print(f"Found credentials for {target}, user: {username}")
        break

if not token:
    print("Could not find GitHub credentials in Credential Manager!")
    sys.exit(1)

# 4. Push with token
remote_url = f"https://{token}@github.com/AIVIETNAM-AIO-MinhHocAyAi/LifeMate-VietASR-Embedding-ZAIC.git"
print("Pushing to origin main...")
push_res = subprocess.run([git_exe, "-C", repo_dir, "push", remote_url, "main"], capture_output=True, text=True)
print(push_res.stdout)
if push_res.stderr:
    # Filter out token from output just in case
    clean_err = push_res.stderr.replace(token, "***")
    print("Push Output:", clean_err)

if push_res.returncode == 0:
    print("[SUCCESS] Successfully committed and pushed to remote GitHub repository!")
else:
    print("[ERROR] Push failed with return code:", push_res.returncode)
