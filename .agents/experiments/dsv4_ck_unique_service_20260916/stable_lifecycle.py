"""Linux ownership keyed to boot ID/start ticks, not wall-clock-derived birth."""
import json
from pathlib import Path
import subprocess
import psutil

def identity(pid):
    text=Path(f'/proc/{pid}/stat').read_text()
    # comm can contain spaces or parentheses; field22 follows the last ')'.
    ticks=int(text.rsplit(')',1)[1].split()[19])
    return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=ticks)

def attach_verified(state):
    proc=psutil.Process(state['pid'])
    pane=int(subprocess.check_output(['tmux','display-message','-p','-t',state['session'],'#{pane_pid}']))
    assert pane==state['pane'] and proc.ppid()==pane
    assert proc.cmdline()==state['command']
    assert proc.cwd()=='/home/pc/Code/sglang'
    assert any(p.path==state['log'] for p in proc.open_files())
    return dict(state,kernel_identity=identity(proc.pid),attachment=dict(
        recorded_birth=state['birth'],observed_birth=proc.create_time(),
        evidence='exact tmux pane,parent PID,command,cwd and open log file'))

def install(life):
    original_start=life.start
    def owned(state):
        assert identity(state['pid'])==state['kernel_identity'],'kernel process identity changed'
        proc=psutil.Process(state['pid']);assert proc.cmdline()==state['command']
        return proc
    def start(label,enabled):
        state=original_start(label,enabled)
        state['kernel_identity']=identity(state['pid'])
        owned(state);life.save(label+'.state.json',state)
        return state
    life.start=start;life.owned=owned
