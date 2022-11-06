import logging
import multiprocessing as mp

import torch
from torchaudio.io import StreamReader

logger = logging.getLogger(__file__)


def audio_stream(queue: mp.Queue):
    """
    Learn more about how to install and use streaming audio here
    https://pytorch.org/audio/stable/tutorials/streaming_api2_tutorial.html
    """

    #streamer = StreamReader(src=":0", format="avfoundation") for mac os
    streamer = StreamReader(src="audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{A5B93F21-D45A-4BB7-B942-66001F04F2DE}", format="dshow") # for windows
    streamer.add_basic_audio_stream(frames_per_chunk=8000, sample_rate=16000)
    stream_iterator = streamer.stream(-1, 1)

    logger.info("Start audio streaming")
    while True:
        (chunk_,) = next(stream_iterator)
        logger.info("Put chunk to queue")
        queue.put(chunk_)


if __name__ == "__main__":
    #model = torch.load("kws.pth").eval()
    model = torch.load("scriptmodule.pt").eval()

    ctx = mp.get_context("spawn")
    chunk_queue = ctx.Queue()
    streaming_process = ctx.Process(target=audio_stream, args=(chunk_queue,))

    streaming_process.start()
    while True:
        try:
            chunk = chunk_queue.get()
            chunk = chunk.view(1, -1)

            with torch.inference_mode():
                result = model(chunk)
            print(f"{chunk.shape=}", result)

            if result > 0.7:
                print("DETECTED KEY WORD")

        except KeyboardInterrupt:
            break
        except Exception as exc:
            raise exc

    streaming_process.join()