use crate::{
    assert_eq_with_tol,
    warps::{MultiBatchMode, SingleBatchMode},
    Precision,
};
use csv::ReaderBuilder;
use std::error::Error;

fn read_csv<T>(file_path: &str) -> Result<Vec<Vec<T>>, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: 'static + Error, // needed to convert parsing error into Box<dyn Error>
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

const WEIGHT_MAX: Precision = 1.0;
fn dtw_weights(len: usize, g: Precision) -> Vec<Precision> {
    let mut weights = vec![0.0; len];
    let half_len = len as Precision / 2.0;
    let e = std::f64::consts::E as Precision;
    for i in 0..len {
        weights[i] = WEIGHT_MAX / (1.0 + e.powf(-g * (i as Precision - half_len)));
    }
    weights
}

#[test]
fn test_device() {
    let (device, _, _, _, _) = crate::utils::get_device();
    println!(
        "Physical device: {:?} \nmax threads per workgroup: {:?}, \nmax compute work group size: {:?}",
        device.physical_device().properties().device_name,
        device.physical_device().properties().max_compute_work_group_size,
        device.physical_device().properties().max_compute_work_group_invocations
    );
}

#[test]
pub fn test_erp() {
    let (device, queue, sba, sda, ma) = crate::utils::get_device();

    let ts = read_csv("tests/data/ts.csv").unwrap();
    let erp_ts: Vec<Vec<Precision>> = read_csv("tests/results/erp.csv").unwrap();

    let gap_penalty = 1.0;
    let start_time = std::time::Instant::now();
    let result = crate::cpu::erp::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &ts,
        &ts,
        gap_penalty,
    );

    println!("GPU ERP time: {:?}", start_time.elapsed());
    for i in 0..ts.len() - 1 {
        for j in i + 1..ts.len() {
            assert_eq_with_tol!(result[i][j], erp_ts[i][j], 1e-6);
        }
    }
}

#[test]
pub fn test_lcss() {
    let (device, queue, sba, sda, ma) = crate::utils::get_device();

    let data = read_csv("tests/data/ts.csv").unwrap();
    let lcss_ts: Vec<Vec<Precision>> = read_csv("tests/results/lcss.csv").unwrap();
    let epsilon = 1.0;
    let start_time = std::time::Instant::now();
    let result = crate::cpu::lcss::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
        epsilon,
    );
    println!("GPU LCSS time: {:?}", start_time.elapsed());
    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], lcss_ts[i][j], 1e-6);
        }
    }
}

#[test]
pub fn test_dtw() {
    let (device, queue, sba, sda, ma) = crate::utils::get_device();

    let data = read_csv("tests/data/ts.csv").unwrap();
    let dtw_ts: Vec<Vec<Precision>> = read_csv("tests/results/dtw.csv").unwrap();
    let start_time = std::time::Instant::now();
    let result = crate::cpu::dtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
    );
    println!("GPU DTW time: {:?}", start_time.elapsed());
    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], dtw_ts[i][j], 1e-6);
        }
    }
}

#[test]
pub fn test_wdtw() {
    let (device, queue, sba, sda, ma) = crate::utils::get_device();

    let g = 0.05;
    let data = read_csv("tests/data/ts.csv").unwrap();
    let wdtw_ts: Vec<Vec<Precision>> = read_csv("tests/results/wdtw.csv").unwrap();
    let start_time = std::time::Instant::now();
    let result = crate::cpu::wdtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
        &dtw_weights(data[0].len(), g),
    );
    println!("GPU WDTW time: {:?}", start_time.elapsed());
    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], wdtw_ts[i][j], 1e-6);
        }
    }
}

#[test]
pub fn test_msm() {
    let (device, queue, sba, sda, ma) = crate::utils::get_device();

    let data = read_csv("tests/data/ts.csv").unwrap();
    let msm_ts: Vec<Vec<Precision>> = read_csv("tests/results/msm.csv").unwrap();
    let start_time = std::time::Instant::now();
    let result = crate::cpu::msm::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
    );
    println!("GPU MSM time: {:?}", start_time.elapsed());
    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], msm_ts[i][j], 1e-6);
        }
    }
}

#[test]
pub fn test_twe() {
    let stiffness = 0.001;
    let penalty = 1.0;
    let data = read_csv("tests/data/ts.csv").unwrap();
    let twe_ts: Vec<Vec<Precision>> = read_csv("tests/results/twe.csv").unwrap();
    let runs = 2;
    let mut avg_time = 0.0;

    for _ in 0..runs {
        let start_time = std::time::Instant::now();
        let (device, queue, sba, sda, sa) = crate::utils::get_device();

        let result = crate::cpu::twe::<MultiBatchMode>(
            device.clone(),
            queue.clone(),
            sba.clone(),
            sda.clone(),
            sa.clone(),
            &data,
            &data,
            stiffness,
            penalty,
        );
        avg_time += start_time.elapsed().as_secs_f64();
        let sum = result
            .iter()
            .map(|v| v.iter().map(|x| *x as f64).sum::<f64>())
            .sum::<f64>();
        let sum_ts = twe_ts
            .iter()
            .map(|v| v.iter().map(|x| *x as f64).sum::<f64>())
            .sum::<f64>();
        println!("SUM (NEW): {}, SUM (OLD): {}", sum, sum_ts);
        for i in 0..data.len() - 1 {
            for j in i + 1..data.len() {
                assert_eq_with_tol!(result[i][j], twe_ts[i][j], 1e-6);
            }
        }
    }
    println!("GPU TWE time: {:?}s", avg_time / runs as f64);
}

#[test]
pub fn test_adtw() {
    let (device, queue, sba, sda, ma) = crate::utils::get_device();
    let warp_penalty = 0.1;
    let data = read_csv("tests/data/ts.csv").unwrap();
    let adtw_ts: Vec<Vec<Precision>> = read_csv("tests/results/adtw.csv").unwrap();
    let start_time = std::time::Instant::now();
    let result = crate::cpu::adtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
        warp_penalty,
    );
    println!("GPU ADTW time: {:?}", start_time.elapsed());

    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], adtw_ts[i][j], 1e-6);
        }
    }
}
