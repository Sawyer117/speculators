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

# exec so CMD becomes PID 1 and receives docker stop signals.
exec "$@"
