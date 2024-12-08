MAIN?=launcher.py
LAUNCH_SCRIPT?=launch.sh
TARGET?=ljk-dao-153.ljk
TARGET_DIR?="~/Documents/escape_time/torch"
TARGET_USER?=azizianw
SOCKET?=./socket.sock

.PHONY:init_ssh
init_ssh:
	eval $(ssh-agent) && ssh-add
	rm -f $(SOCKET)
	ssh -M -S $(SOCKET) -o ControlPersist=60m $(TARGET_USER)@$(TARGET) exit

.PHONY:deploy
deploy:
	rsync -Pavu -e "ssh -S $(SOCKET)"  --include='*.py' --include='configs/*.yaml' --include="*.sh" --include="*/" --include=".project-root" --prune-empty-dirs --exclude='*' . $(TARGET_USER)@$(TARGET):$(TARGET_DIR)

.PHONY:connect
connect:
	ssh -S $(SOCKET) -t $(TARGET_USER)@$(TARGET) "mkdir -p $(TARGET_DIR) && cd $(TARGET_DIR) && bash --login"

.PHONY:clear_logs
clear_logs:
	trash -f ./logs/*
	ssh -S $(SOCKET) $(TARGET_USER)@$(TARGET) "rm -rf $(TARGET_DIR)/logs/*"

.PHONY:fetch_logs
fetch_logs:
	rsync -Pavu -e "ssh -S $(SOCKET)" $(TARGET_USER)@$(TARGET):$(TARGET_DIR)/logs/* ./logs/

.PHONY:analyze
analyze:fetch_logs
	python analyze.py

format:
	pre-commit run -a

launch:deploy
	ssh -S $(SOCKET) $(TARGET_USER)@$(TARGET) "cd $(TARGET_DIR) && bash $(LAUNCH_SCRIPT)"
