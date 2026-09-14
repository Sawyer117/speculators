#!/bin/bash
# LingShu entrypoint: generate SSH host keys if missing, start sshd, then hand over to CMD.
# Requirement: the container must STAY RUNNING so Clab can exec into it.
set -e

if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    echo "SSH host keys not found. Generating..."
    ssh-keygen -A
fi

echo "Starting SSH daemon..."
/usr/sbin/sshd

# Put conda+CANN on the path for whatever CMD runs, so a non-interactive
# `docker exec <c> python -c "import torch"` works without a login shell.
# shellcheck disable=SC1091
[ -f /etc/profile.d/00-dsv4.sh ] && . /etc/profile.d/00-dsv4.sh

# exec so CMD becomes PID 1 and receives docker stop signals.
exec "$@"
