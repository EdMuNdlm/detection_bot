import signal
import time

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('echo', None, 'Text to echo.')

interruppted = False
def handle(*args):
    logging.info('Signal catched. Exiting.')
    global interruppted
    interruppted = True

def main(argv):
    del argv  # Unused
    logging.info('This is a docker-python program.')

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    while not interruppted:
        logging.info(f'Echo: {FLAGS.echo}')
        time.sleep(1)

    logging.info('Done.')

if __name__ == '__main__':
    app.run(main)