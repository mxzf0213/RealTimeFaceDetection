//console.log('Loading Model...');
async function loadModel() {
    const model = await tf.loadLayersModel('./model/model.json')
    return model
}

const videoWidth = 600
const videoHeight = 500

const new_width = 224
const new_height = 224

const truth_thresh = 0.6
const nms_thresh = 0.3

const anchors = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]];

function isAndroid() {
    return /Android/i.test(navigator.userAgent);
}

function isiOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
    return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;

    const mobile = isMobile();
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: mobile ? undefined : videoWidth,
            height: mobile ? undefined : videoHeight,
        },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();

    return video;
}


function preprocess(imgData) {
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(imgData).toFloat()
        // Normalize the image 
        const resized = tf.image.resizeBilinear(tensor, [new_width, new_height])
        const normalized = resized.div(tf.scalar(255.0))
        const batched = normalized.expandDims(0)
        return batched
    })
}

async function predict(net, ctx) {
    imgData = ctx.getImageData(0, 0, video.width, video.height)
    const pred = net.predict(preprocess(imgData)).dataSync()
    return pred
}

function sigmoid(x) {
    return 1. / (1 + Math.exp(-x));
}

function iou(a, b) {
    const ixmin = Math.max(a[1], b[1])
    const ixmax = Math.min(a[1] + a[3], b[1] + b[3]);
    const iymin = Math.max(a[2], b[2])
    const iymax = Math.min(a[2] + a[4], b[2] + b[4]);
    if (ixmin > ixmax || iymin > iymax) return 0;

    const iarea = (iymax - iymin) * (ixmax - ixmin);
    const oarea = -iarea + a[4] * a[3] + b[4] * b[3];
    return 1. * iarea / oarea;
}

async function processpreds(preds) {
    let faces = []
    var cnt = 0
    var index = 0
    for (var i = 0; i < preds.length; i += 30) {
        const row = Math.floor(index / 7)
        const col = index % 7
        for (var j = 0; j < 5; j++) {
            var box_conf = sigmoid(preds[i + j * 6 + 4])
            if (box_conf < truth_thresh) continue;
            var box_x = (sigmoid(preds[i + j * 6 + 0]) + col) / 7.0;
            var box_y = (sigmoid(preds[i + j * 6 + 1]) + row) / 7.0;
            var box_w = Math.exp(preds[i + j * 6 + 2]) * anchors[j][0] / 7.0;
            var box_h = Math.exp(preds[i + j * 6 + 3]) * anchors[j][1] / 7.0;
            box_x = (box_x - box_w / 2) * videoWidth;
            box_y = (box_y - box_h / 2) * videoHeight;
            box_w = box_w * videoWidth;
            box_h = box_h * videoHeight;
            if (box_x < 0) box_x = 0
            if (box_y < 0) box_y = 0
            if (box_x + box_w > videoWidth - 1) box_w = videoWidth - 1 - box_x;
            if (box_y + box_h > videoHeight - 1) box_h = videoHeight - 1 - box_y;
            faces[cnt++] = [box_conf, box_x, box_y, box_w, box_h];
        }
        index = index + 1
    }
    //nms_sort
    faces.sort(function (a, b) { return a[0] >= b[0]; });
    for (var i = 0; i < faces.length; i++) {
        if (faces[i][0] == 0) continue;
        for (var j = i + 1; j < faces.length; j++) {
            if (faces[j][0] == 0) continue;
            if (iou(faces[i], faces[j]) >= nms_thresh) {
                faces[j][0] = 0;
            }
        }
    }
    return faces;
}

async function drawFaceRec(faces, ctx) {
    ctx.lineWidth = "3";
    ctx.strokeStyle = 'red';
    ctx.font = '20px Arial';
    for (var i = 0; i < faces.length; i++) {
        if (faces[i][0] == 0) continue;
        ctx.fillStyle = 'red';
        ctx.fillRect(faces[i][1], faces[i][2] - 30, 120, 30);
        ctx.fillStyle = 'black';
        ctx.fillText("Person: " + String((faces[i][0] * 100).toFixed(0)) + "%", faces[i][1], faces[i][2] - 10);
        ctx.strokeRect(faces[i][1], faces[i][2], faces[i][3], faces[i][4]);
    }
}

function detectFaceInRealTime(video, net) {
    const canvas = document.getElementById('output');
    const ctx = canvas.getContext('2d');

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    async function faceDetectionFrame() {
        ctx.clearRect(0, 0, videoWidth, videoHeight);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
        ctx.restore();
        let preds = []
        preds = await predict(net, ctx)
        let faces = await processpreds(preds)
        await drawFaceRec(faces, ctx)
        requestAnimationFrame(faceDetectionFrame);
    }
    faceDetectionFrame();
}

async function bindPage() {
    const faceModel = await loadModel()

    //console.log('Model loaded!')
    //faceModel.summary()

    let video;

    try {
        video = await loadVideo();
    } catch (e) {
        let info = document.getElementById('info');
        info.textContent = 'this browser does not support video capture,' +
            'or this device does not have a camera';
        info.style.display = 'block';
        throw e;
    }

    detectFaceInRealTime(video, faceModel);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

bindPage();