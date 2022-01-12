const MODELS = {
    NONE: { label: "None", index: 0 },
    CNN_7: { label: "CNN_7", index: 1 },
    CNN_4: { label: "CNN_4", index: 2 },
    RF: { label: "Random forest", index: 3 },
    SVM: { label: "Super vetor machine", index: 4 }
}

const video = document.getElementById('video');
const video_placeholder = document.getElementById('video-placeholder');
const canvas = document.getElementById('canvas');
const downloadLink = document.getElementById('downloadLink');
const currentModelUI = document.getElementById('current-model');
const loading = document.getElementById('loading');


let isPredict = false;
let modelUsing = MODELS.NONE;

function getModelInfo(index) {
    for (model in MODELS) {
        if (MODELS[model].index === index) {
            return MODELS[model];
        }
    }
    return MODELS.NONE;
}

function changeModel(index) {
    modelUsing = getModelInfo(index);
}

function updateModelUI() {
    let index = modelUsing.index;
    index++;
    if (index > (Object.keys(MODELS).length - 1)) index = 0;
    changeModel(index);
    currentModelUI.innerHTML = modelUsing.label;

}

function toggleLoading() {
    loading.classList.remove('display-none');
    setTimeout(() => {
        loading.classList.add('display-none');
    }, 3000);
}

function handleClickPredict() {
    updateModelUI();
    toggleLoading();
    if (modelUsing.index === MODELS.CNN_7.index) {
        video.src = "/video_detect_stream_CNN_7";
    } else if (modelUsing.index === MODELS.CNN_4.index) {
        video.src = "/video_detect_stream_CNN_4";
    }
    else if (modelUsing.index === MODELS.RF.index) {
        video.src = "/video_detect_stream_RF";
    } else if (modelUsing.index === MODELS.SVM.index) {
        video.src = "/video_detect_stream_SVM";
    } else {
        video.src = "/video_normal_stream";
    }
}

function handleCaptureVideo() {
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    let image_data_url = canvas.toDataURL('image/jpeg');
    downloadLink.href = image_data_url;
    downloadLink.click();
}
