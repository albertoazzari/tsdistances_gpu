use crate::{assert_eq_with_tol, warps::{MultiBatchMode, SingleBatchMode}};
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

#[test]
fn test_device() {
    let (device, _, _, _) = crate::utils::get_device();
    println!(
        "Physical device: {:?} type: {:?}",
        device.physical_device().properties().device_name,
        device.physical_device().properties().device_name
    );
}
#[test]
pub fn test_erp() {
    let (device, queue, sba, sda) = crate::utils::get_device();

    let ts = read_csv("tests/data/ts.csv").unwrap();
    let erp_ts: Vec<Vec<f64>> = read_csv("tests/data/erp_ts.csv").unwrap();

    let gap_penalty = 1.0;
    let result = crate::cpu::erp::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        &ts,
        &ts,
        gap_penalty,
    );

    for i in 0..ts.len()-1 {
        for j in i+1..ts.len() {
            assert_eq_with_tol!(result[i][j], erp_ts[i][j], 1e-6);
        }
    }
}

#[test]
pub fn test_lcss() {
    let (device, queue, sba, sda) = crate::utils::get_device();

    let data = read_csv("tests/data/ts.csv").unwrap();

    let espilon = 1.0;
    let result = crate::cpu::lcss::<SingleBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        &data[0],
        &data[1],
        espilon,
    );

    assert_eq!(result, 0.752, "Failed to compute LCSS distance");
}
