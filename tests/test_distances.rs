use std::fmt::write;

use csv::ReaderBuilder;
use tsdistances_gpu::{
    cpu::{adtw, dtw, erp, lcss, msm, twe, wdtw},
    utils::get_device,
    warps::{GpuBatchMode, MultiBatchMode, SingleBatchMode},
};

fn read_txt<T>(file_path: &str) -> Result<Vec<Vec<T>>, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: 'static + std::error::Error,
{
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(if file_path.ends_with(".tsv") { b'\t' } else { b',' })
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

pub fn write_csv<T>(file_path: &str, data: &[Vec<T>]) -> Result<(), Box<dyn std::error::Error>>
where
    T: std::fmt::Display,
{
    let mut wtr = csv::Writer::from_path(file_path)?;
    for row in data {
        wtr.write_record(row.iter().map(|item| item.to_string()))?;
    }
    wtr.flush()?;
    Ok(())
}

const WEIGHT_MAX: f32 = 1.0;
fn dtw_weights(len: usize, g: f32) -> Vec<f32> {
    let mut weights = vec![0.0; len];
    let half_len = len as f32 / 2.0;
    let e = std::f64::consts::E as f32;
    for i in 0..len {
        weights[i] = WEIGHT_MAX / (1.0 + e.powf(-g * (i as f32 - half_len)));
    }
    weights
}

#[test]
fn test_erp_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

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
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

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
    let ds_name = "CBF";
    let train_data: Vec<Vec<f32>> = read_txt(&format!("../../DATA/ucr/{}/{}_TRAIN.tsv", ds_name, ds_name)).unwrap();
    let test_data: Vec<Vec<f32>> = read_txt(&format!("../../DATA/ucr/{}/{}_TEST.tsv", ds_name, ds_name)).unwrap();
    // let train_data: Vec<Vec<f32>> = read_txt("../../DATA/ucr/NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TRAIN.tsv").unwrap();
    // let test_data: Vec<Vec<f32>> = read_txt("../../DATA/ucr/NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TEST.tsv").unwrap();

    // let start = std::time::Instant::now();

    // let (device, queue, sba, sda, ma) = get_device();

    // println!("Device elapsed time: {:?}", start.elapsed());
    // for i in 0..train_data.len() {
    //     for j in 0..test_data.len() {
            
    //         let result = dtw::<SingleBatchMode>(
    //             device.clone(),
    //             queue.clone(),
    //             sba.clone(),
    //             sda.clone(),
    //             ma.clone(),
    //             &train_data[i],
    //             &test_data[j],
    //         );
    //     }
    // }
    // println!("Single DTW elapsed time: {:?}", start.elapsed());


    let start = std::time::Instant::now();

    let (device, queue, sba, sda, ma) = get_device();

    println!("Device elapsed time: {:?}", start.elapsed());

    let result = dtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
    );
    println!("DTW elapsed time: {:?}", start.elapsed());

    write_csv(&format!("tests/{}_DTW_TE.csv", ds_name), &result).unwrap();
    // Device elapsed time: 278.698142ms
    // Single DTW elapsed time: 23.71971498s
    // Device elapsed time: 213ns
    // DTW elapsed time: 12.042854409s

}

#[test]
fn test_wdtw_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

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
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

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
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

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
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

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
