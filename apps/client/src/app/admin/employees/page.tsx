'use client';

import { useCallback, useEffect, useState, type ChangeEvent } from 'react';
import Image from 'next/image';

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000'
).replace(/\/$/, '');

type Employee = {
  id: number;
  name: string;
  email?: string | null;
  department?: string | null;
  role?: string | null;
  location?: string | null;
  has_photo: boolean;
};

export default function EmployeeAdminPage() {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [uploadingId, setUploadingId] = useState<number | null>(null);

  const fetchEmployees = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE_URL}/api/employees/`);
      if (!response.ok) {
        throw new Error('Failed to fetch employee list');
      }
      const data = (await response.json()) as Employee[];
      setEmployees(data);
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.message
          : 'Unknown error while loading employees';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchEmployees();
  }, [fetchEmployees]);

  const handlePhotoUpload = async (
    employeeId: number,
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.type.startsWith('image/')) {
      setError('Please choose a valid image file.');
      event.target.value = '';
      return;
    }

    try {
      setUploadingId(employeeId);
      setError(null);
      setSuccessMessage(null);
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(
        `${API_BASE_URL}/api/employees/${employeeId}/photo`,
        {
        method: 'POST',
        body: form
        }
      );
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail ?? 'Upload failed');
      }
      await fetchEmployees();
      setSuccessMessage('Employee photo uploaded successfully.');
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Photo upload failed';
      setError(message);
    } finally {
      setUploadingId(null);
      event.target.value = '';
    }
  };

  return (
    <main className="mx-auto max-w-6xl p-6">
      <h1 className="mb-2 text-2xl font-bold text-gray-900">Employee Photos</h1>
      <p className="mb-6 text-sm text-gray-600">
        Upload and manage reference photos for face verification.
      </p>

      {error && (
        <div className="mb-4 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {successMessage && (
        <div className="mb-4 rounded border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-700">
          {successMessage}
        </div>
      )}

      {loading ? (
        <div className="text-sm text-gray-600">Loading employees...</div>
      ) : (
        <div className="overflow-x-auto rounded-lg border bg-white shadow-sm">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 text-left text-gray-700">
              <tr>
                <th className="px-4 py-3">Employee</th>
                <th className="px-4 py-3">Department</th>
                <th className="px-4 py-3">Photo</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Action</th>
              </tr>
            </thead>
            <tbody>
              {employees.map((employee) => (
                <tr key={employee.id} className="border-t">
                  <td className="px-4 py-3">
                    <div className="font-medium text-gray-900">
                      {employee.name}
                    </div>
                    <div className="text-xs text-gray-500">
                      {employee.email ?? '-'}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-gray-700">
                    {employee.department || employee.role || '-'}
                  </td>
                  <td className="px-4 py-3">
                    {employee.has_photo ? (
                      <Image
                        src={`${API_BASE_URL}/api/employees/${employee.id}/photo`}
                        alt={`${employee.name} profile`}
                        className="h-14 w-14 rounded object-cover ring-1 ring-gray-200"
                        width={56}
                        height={56}
                        unoptimized
                      />
                    ) : (
                      <div className="flex h-14 w-14 items-center justify-center rounded bg-gray-100 text-xs text-gray-400">
                        No photo
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded px-2 py-1 text-xs font-semibold ${
                        employee.has_photo
                          ? 'bg-green-100 text-green-700'
                          : 'bg-amber-100 text-amber-700'
                      }`}
                    >
                      {employee.has_photo ? 'Photo set' : 'Missing'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <label className="inline-flex cursor-pointer items-center rounded bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700">
                      {uploadingId === employee.id
                        ? 'Uploading...'
                        : 'Upload Photo'}
                      <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        disabled={uploadingId === employee.id}
                        onChange={(event) =>
                          void handlePhotoUpload(employee.id, event)
                        }
                      />
                    </label>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <section className="mt-6 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="text-base font-semibold text-gray-900">
          When to upload photos
        </h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-gray-600">
          <li>Upload right after creating a new employee profile.</li>
          <li>Re-upload when appearance changes or recognition quality drops.</li>
          <li>Use this page only for employees, not visitor check-ins.</li>
        </ul>
      </section>
    </main>
  );
}
