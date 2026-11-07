"""
Script hỗ trợ đồng bộ và đẩy mã nguồn lên GitHub của bạn:
https://github.com/AIVIETNAM-AIO-MinhHocAyAi
"""

import os
import subprocess
import sys

GITHUB_REPO_URL = "https://github.com/AIVIETNAM-AIO-MinhHocAyAi/LifeMate-VietASR-Embedding-ZAIC.git"

def check_git_installed():
    try:
        res = subprocess.run(["git", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0
    except FileNotFoundError:
        return False

def push_project_to_github():
    print("=" * 60)
    print(" CHUẨN BỊ ĐẨY DỰ ÁN LÊN GITHUB")
    print(f" Target Repository: {GITHUB_REPO_URL}")
    print("=" * 60)

    if not check_git_installed():
        print("\n[CẢNH BÁO] Không tìm thấy lệnh 'git' trong PATH của Windows.")
        print("Bạn có thể thực hiện theo một trong các cách sau:")
        print("1. Nếu máy đã cài Git Bash: mở Git Bash tại thư mục này và chạy các lệnh:")
        print("   git init")
        print("   git add .")
        print('   git commit -m "feat: init LifeMate ASR under 1.5B and BGE-M3 pipeline"')
        print(f"   git remote add origin {GITHUB_REPO_URL}")
        print("   git branch -M main")
        print("   git push -u origin main")
        print("\n2. Hoặc upload trực tiếp thư mục này lên GitHub web interface.")
        return

    commands = [
        ["git", "init"],
        ["git", "add", "."],
        ["git", "commit", "-m", "feat: complete LifeMate ASR Whisper Turbo under 1.5B and BGE-M3 training pipeline"],
        ["git", "branch", "-M", "main"],
        ["git", "remote", "add", "origin", GITHUB_REPO_URL],
        ["git", "push", "-u", "origin", "main"]
    ]

    for cmd in commands:
        cmd_str = " ".join(cmd)
        print(f"[EXEC] {cmd_str}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            # Nếu remote đã tồn tại, cập nhật lại remote set-url
            if "remote origin already exists" in res.stderr:
                subprocess.run(["git", "remote", "set-url", "origin", GITHUB_REPO_URL])
            else:
                print(f"[INFO] {res.stderr.strip()}")
        else:
            if res.stdout.strip():
                print(res.stdout.strip())

    print("\n[SUCCESS] Hoàn tất tiến trình Git.")

if __name__ == "__main__":
    push_project_to_github()
