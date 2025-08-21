## Activate Environment
```
source /Users/arnocombrinck/Library/Caches/pypoetry/virtualenvs/sae-ppa-chatbot-QSaLMZ4a-py3.11/bin/activate
```

## Add/Remove New Python Package to Poetry
```
poetry add/remove <package name>
```

## Run FastAPI Server
```
uvicorn fastapi-server.app:app --reload
```

## Locally Zip Project Folder
```
zip -r sae-ppa-chatbot.zip /path/to/folder
```

## Transfer Zip File to EC2 VM
```
rsync -avz -e "ssh -i '/Users/arnocombrinck/AWS SSH Keys/ubuntu-flowise-ssh-key-pair.pem'" sae-ppa-chatbot.zip ubuntu@flowise-vm:/home/ubuntu/
```

## EC2 VM Unzip Project File
```
unzip sae-ppa-chatbot.zip
```

## Install Python Packages with Poetry
```
poetry install
```

## Run FastAPI Server (from /home/ubuntu)
```
pm2 start "poetry run uvicorn fastapi-server.app:app --app-dir /home/ubuntu/sae-ppa-chatbot" --name sae-ppa-chatbot --interpreter python3
```

## Ensure App Will Restart Between Reboots
```
pm2 startup
pm2 save
```
