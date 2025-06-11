use csv::ReaderBuilder;
use tsdistances_gpu::{
    cpu::{adtw, dtw, erp, lcss, msm, twe, wdtw},
    utils::get_device,
    warps::{GpuBatchMode, MultiBatchMode},
    Float,
};

fn read_csv<T>(file_path: &str) -> Result<Vec<Vec<T>>, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: 'static + std::error::Error,
{
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row: Vec<T> = record
            .iter()
            .map(|s| s.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        records.push(row);
    }
    Ok(records)
}

const WEIGHT_MAX: Float = 1.0;
fn dtw_weights(len: usize, g: Float) -> Vec<Float> {
    let mut weights = vec![0.0; len];
    let half_len = len as Float / 2.0;
    let e = std::f64::consts::E as Float;
    for i in 0..len {
        weights[i] = WEIGHT_MAX / (1.0 + e.powf(-g * (i as Float - half_len)));
    }
    weights
}

#[test]
fn test_erp_distance() {
    let train_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let (device, queue, sba, sda, ma) = get_device();

    let gap_penalty = 1.0;

    let result = erp::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        gap_penalty,
    );
}

#[test]
fn test_lcss_distance() {
    let train_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let (device, queue, sba, sda, ma) = get_device();
    let epsilon = 1.0;
    let start = std::time::Instant::now();
    let result = lcss::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        epsilon,
    );
    let elapsed = start.elapsed();
    println!("LCSS elapsed time: {:?}", elapsed);
}

#[test]
fn test_dtw_distance() {
    let train_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let (device, queue, sba, sda, ma) = get_device();

    let result = dtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
    );
}

#[test]
fn test_wdtw_distance() {
    let train_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let (device, queue, sba, sda, ma) = get_device();
    let g = 0.05;
    let weights = dtw_weights(train_data[0].len(), g);

    let result = wdtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        &weights,
    );
}

#[test]
fn test_adtw_distance() {
    let train_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let (device, queue, sba, sda, ma) = get_device();

    let w = 0.1;

    let result = adtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        w,
    );
}

#[test]
fn test_msm_distance() {
    let train_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let (device, queue, sba, sda, ma) = get_device();

    let result = msm::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
    );
}

#[test]
fn test_twe_distance() {
    let train_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<Float>> = read_csv("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let (device, queue, sba, sda, ma) = get_device();

    let stiffness = 0.001;
    let penalty = 1.0;

    let result = twe::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        stiffness,
        penalty,
    );
}
