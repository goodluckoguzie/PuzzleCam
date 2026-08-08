# PuzzleCam: How to use

Two ways to run: **browser** (phones OK) or **Python** (laptop webcam only).

---

## Browser mode

### Open this URL (important)

Use one of these in your browser:

- **http://localhost:5500/**
- **http://127.0.0.1:5500/**

Do **not** open `http://0.0.0.0:5500/`.

`0.0.0.0` is only the server bind address (listen on all interfaces). Chrome and most browsers show an error such as **ERR_ADDRESS_INVALID** if you put `0.0.0.0` in the address bar.

Hard refresh after changes: **Ctrl+Shift+R**

### Start the browser server

```bash
cd /home/goodluck/Desktop/MyProjects/Tutorial/PuzzleCam
python3 -m http.server 5500
```

Then open **http://localhost:5500/** (not 0.0.0.0).

Allow the camera when the browser asks. You need internet the first time (MediaPipe loads from a CDN).

---

## Python mode (laptop)

OpenCV window with the same gesture flow. No browser or phone needed.

```bash
cd /home/goodluck/Desktop/MyProjects/Tutorial/PuzzleCam
conda activate home
python puzzle_cam.py
```

Or install deps once: `pip install -r requirements.txt`

Keys: `q` quit, `r` reset, `d` download strip (after 3 saves), `m` toggle mirror.

Saved images go to `output/`.

---

## Take a photo

This is the **original** capture style: frame the photo with **both hands**.

![PuzzleCam how to use](puzzlecam-how-to.png)

1. Allow the camera
2. Wait until status says **ready** or **looking for hands...**
3. Raise **both hands** so the camera sees them
4. On **each hand**, pinch thumb + index finger together
5. Spread your hands apart to make the yellow box bigger
6. Keep both pinches held until countdown starts
7. Photo is taken automatically

Tips for a bigger box:
- Move hands farther apart (left/right and up/down)
- Keep both pinches active
- Stay in good light

## Solve the puzzle

1. Photo becomes a 3x3 black-and-white puzzle
2. Pinch one hand on a piece to pick it up
3. Move it, then release the pinch to drop it
4. Place all 9 pieces correctly

## Save

**Browser:** when you see **COMPLETE! Fist to save**, hold a closed fist. Puzzle saves to the Strip. After 3 saves, click **download strip**.

**Python:** same fist gesture when complete. Press `d` after 3 saves to write a strip PNG to `output/`.

## Gestures summary

| Gesture | Action |
|---------|--------|
| Both hands pinch | Make the capture frame |
| Hold both pinches | Start countdown + take photo |
| One-hand pinch on a piece | Drag puzzle tile |
| Hold closed fist | Save puzzle / reset |
